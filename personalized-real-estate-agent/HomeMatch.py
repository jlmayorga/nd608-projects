import logging
import pickle
import random
import time
import uuid
from functools import partial
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple

import gradio as gr
import lancedb
import requests
import torch
from PIL import Image
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, computed_field, NonNegativeInt, NonNegativeFloat
from transformers import CLIPModel, CLIPProcessor, PreTrainedTokenizer, PreTrainedTokenizerBase, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PICKLE_FILE = Path("listings.pickle")
IMAGE_DIR = Path("images")
LANCEDB_PATH = Path("data/lancedb")
TABLE_NAME = "RE_LISTINGS"
CLIP_MODEL = "openai/clip-vit-large-patch14"


class RealEstateListing(BaseModel):
    title: str
    neighborhood: str
    city: str
    state: str
    price: NonNegativeInt
    bedrooms: NonNegativeInt
    bathrooms: NonNegativeFloat
    property_type: str
    size: NonNegativeFloat
    description: str
    neighborhood_description: str
    amenities: list[str]
    image_url: str
    image_bytes: bytes

    @computed_field
    def as_str(self) -> str:
        return ""

    @computed_field
    def formatted_price(self) -> str:
        return "${:,}".format(self.price)

    @computed_field
    def formatted_size(self) -> str:
        return "{:,} sqft".format(self.size)

    @property
    def image_as_pil(self):
        return Image.open(BytesIO(self.image_bytes))


class RealEstateListings(BaseModel):
    listings: list[RealEstateListing]


class RealEstateListingLanceDB(LanceModel):
    title: str
    neighborhood: str
    city: str
    state: str
    price: NonNegativeInt
    bedrooms: NonNegativeInt
    bathrooms: NonNegativeFloat
    property_type: str
    size: NonNegativeFloat
    description: str
    neighborhood_description: str
    amenities: list[str]
    image_url: str
    image_bytes: bytes
    vector: Vector(768)

    @property
    def image_as_pil(self):
        return Image.open(BytesIO(self.image_bytes))

    def __str__(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Location: {self.neighborhood}, {self.city}, {self.state}\n"
            f"Price: ${self.price:,}\n"
            f"Property Type: {self.property_type}\n"
            f"Size: {self.size:,} sqft\n"
            f"Bedrooms: {self.bedrooms}\n"
            f"Bathrooms: {self.bathrooms}\n"
            f"Amenities: {', '.join(self.amenities)}\n"
            f"Description: {self.description}\n"
            f"Neighborhood: {self.neighborhood_description}"
        )


def generate_listings() -> RealEstateListings:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful real estate agent"),
        ("user", dedent('''
    Generate 15 real estate listings using the following example as a reference:
    ###
    Title: Green Oaks Stunning Residence
    Neighborhood: Green Oaks
    City: Austin
    State: Texas
    Price: 800000
    Bedrooms: 3
    Bathrooms: 2
    Property Type: House
    Size: 2000
    
    Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.
    
    Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
    ###
    
    Include amenities in the listings, like pool, backyard, outdoor kitchen, vegetable garden, fireplace, firepit, etc..
        
    {format_instructions}
    '''))
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    parser = PydanticOutputParser(pydantic_object=RealEstateListings)
    chain = prompt | llm | parser
    response = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
    })

    generated_listings = response.listings

    listings_with_images = []
    logger.info("Generating Images...")
    for listing in generated_listings:
        prompt = dedent(
            f"""
                Photo of {listing.description}.
                1/100s, ISO 100, Daylight
                """
        )
        image_url = DallEAPIWrapper(size='256x256').run(prompt)
        listing.image_url = image_url
        img_response = requests.get(image_url)
        listing.image_bytes = img_response.content

        with open(f"images/{uuid.uuid4()}.png", "wb") as f:
            f.write(img_response.content)

        listings_with_images.append(listing)
        # Sleep to avoid rate limiting
        time.sleep(15)

    response.listings = listings_with_images

    return response


def get_listings() -> list[RealEstateListing]:
    if PICKLE_FILE.exists():
        logger.info("Loading existing listings...")
        with PICKLE_FILE.open("rb") as f:
            listings = pickle.load(f)
    else:
        logger.info("Generating new listings...")
        listings = generate_listings()
        with PICKLE_FILE.open("wb") as f:
            pickle.dump(listings, f)

    logger.info(f"Fetched {len(listings.listings)} listings")
    return listings.listings


def find_listing(
        table: lancedb.table.Table,
        model: CLIPModel,
        tokenizer: PreTrainedTokenizerBase,
        description: str,
        bedrooms: int,
        bathrooms: float,
        budget: float,
        amenities: List[str]
) -> RealEstateListingLanceDB:
    client_preferences = (f"Property with {bedrooms} bedrooms and {bathrooms} bathrooms. "
                          f" With amenities: {",".join(amenities)}")

    inputs = tokenizer(client_preferences, padding=True, truncation=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)[0].cpu().detach().numpy()

    query = table.search(text_features)

    listings = query.to_pydantic(RealEstateListingLanceDB)

    return random.choice(listings) if listings else None


def enhance_listing_description(listing: RealEstateListingLanceDB, user_preferences: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful real estate agent. Your task is to enhance the description of a property based on the user's preferences without changing any factual information about the property."),
        ("user", dedent(f'''
        Property details:
        {str(listing)}

        User preferences:
        {user_preferences}

        Enhance the property description to appeal to the user's preferences without changing any factual information about the property. Keep the enhanced description concise, focusing on how the property meets the user's needs.
        '''))
    ])

    chain = prompt | llm
    response = chain.invoke({})

    return response.content


def get_recommendation(
        table: lancedb.table.Table,
        model: CLIPModel,
        tokenizer: PreTrainedTokenizer,
        description: str,
        bedrooms: int,
        bathrooms: float,
        budget: float,
        amenities: List[str]
) -> Tuple[Optional[Image.Image], str]:
    listing = find_listing(table, model, tokenizer, description, bedrooms, bathrooms,
                           budget, amenities)
    if listing is None:
        return None, "We couldn't find any property matching the criteria, please try again"

    user_preferences = (f"Looking for a property with {bedrooms} bedrooms, "
                        f"{bathrooms} bathrooms, and a budget of ${budget}. "
                        f"Desired amenities: {', '.join(amenities)}. Additional preferences: {description}")

    enhanced_description = enhance_listing_description(listing, user_preferences)
    return listing.image_as_pil, enhanced_description


def get_listing_embeddings(listing: RealEstateListing) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Enhance with other listing properties
    text_description = (
        f"{listing.title}. {listing.property_type} in {listing.neighborhood}, {listing.city}, {listing.state}. "
        f"{listing.bedrooms} bedrooms, {listing.bathrooms} bathrooms, {listing.size} sqft. "
        f"Price: ${listing.price}. Amenities: {', '.join(listing.amenities)}. "
        f"{listing.description} {listing.neighborhood_description}")

    processor_output = processor(

        text=[text_description],
        images=listing.image_as_pil,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    image = processor_output["pixel_values"].to(device)
    image_embeddings = model.get_image_features(image)

    return image_embeddings[0].cpu()


def generate_embeddings(real_estate_listings: list[RealEstateListing], table: lancedb.table.Table):
    table.add([
        RealEstateListingLanceDB(
            **listing.model_dump(),
            vector=get_listing_embeddings(listing).detach().numpy()
        )
        for listing in real_estate_listings
    ])


def main():
    """Main function."""
    load_dotenv()
    logger.info("Fetching listings...")
    listings = get_listings()

    logger.info("Generating embeddings...")
    db = lancedb.connect(LANCEDB_PATH)
    db.drop_table(TABLE_NAME, ignore_missing=True)
    table = db.create_table(TABLE_NAME, schema=RealEstateListingLanceDB.to_arrow_schema())

    generate_embeddings(listings, table)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)

    logger.info("Reticulating splines...")
    cities = sorted(set(listing.city for listing in listings))
    states = sorted(set(listing.state for listing in listings))
    property_types = sorted(set(listing.property_type for listing in listings))
    amenities = sorted(set(amenity for listing in listings for amenity in listing.amenities))
    min_price, max_price = min(listing.price for listing in listings), max(
        listing.price for listing in listings) if listings else (None, None)
    min_bedrooms, max_bedrooms = min(listing.bedrooms for listing in listings), max(
        listing.bedrooms for listing in listings) if listings else (None, None)
    min_bathrooms, max_bathrooms = min(listing.bathrooms for listing in listings), max(
        listing.bathrooms for listing in listings) if listings else (None, None)

    demo = gr.Interface(
        title="🏠 HomeMatch 🔥",
        description="A smart and helpful Real Estate Agent",
        allow_flagging="never",
        fn=partial(get_recommendation, table, model, tokenizer),
        inputs=[
            gr.Text(label="Description", placeholder="Description"),
            gr.Slider(label="Bedrooms", minimum=min_bedrooms, maximum=max_bedrooms, value=min_bedrooms, step=1),
            gr.Slider(label="Bathrooms", minimum=min_bathrooms, maximum=max_bathrooms, value=min_bathrooms, step=0.5),
            gr.Slider(label="Budget", maximum=max_price, minimum=min_price, value=min_price),
            gr.CheckboxGroup(label="Amenities", choices=amenities),
        ],
        outputs=[gr.Image(label="Image", show_download_button=False, height=256, width=256),
                 gr.Text(label="Description")],
    )

    demo.launch()


if __name__ == "__main__":
    main()
