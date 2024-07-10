
## Overview
HomeMatch is a smart real estate agent application that uses AI to generate and recommend real estate listings based on user preferences. It utilizes several AI models and technologies, including OpenAI's GPT for text generation, DALL-E for image generation, and CLIP for embedding generation and similarity search.
## Key Components
### Dependencies

- gradio: For creating the user interface
- lancedb: For vector database operations
- torch and transformers: For working with the CLIP model
- langchain: For integrating with OpenAI's models
- pydantic: For data validation and settings management

### Main Classes

- RealEstateListing: Pydantic model for individual real estate listings
- RealEstateListings: Pydantic model for a collection of listings
- RealEstateListingLanceDB: LanceDB model for storing listings with embeddings

### Key Functions

- generate_listings(): Generates new real estate listings using OpenAI's GPT
- get_listings(): Retrieves or generates listings
- find_listing(): Searches for a matching listing based on user preferences
- get_recommendation(): Main function to get a property recommendation
- get_listing_embeddings(): Generates embeddings for a listing using CLIP
- generate_embeddings(): Generates embeddings for all listings
- main(): Entry point of the application

### Workflow

- The application starts by loading or generating real estate listings.
- Listings are stored in a LanceDB database along with their CLIP embeddings.
- When a user inputs their preferences, the application:
  - Encodes the preferences using CLIP
  - Searches the LanceDB for similar listings
  - Selects a matching listing
  - Enhances the listing description (TODO)
  - Returns the listing image and description to the user

```mermaid
sequenceDiagram
    participant User
    participant GradioInterface
    participant MainFunction
    participant ListingGenerator
    participant LanceDB
    participant CLIP
    participant DallE
    participant OpenAI

    User->>GradioInterface: Enter search criteria
    GradioInterface->>MainFunction: Call get_recommendation
    MainFunction->>ListingGenerator: get_listings()
    alt Listings exist
        ListingGenerator-->>MainFunction: Return cached listings
    else Listings don't exist
        ListingGenerator->>OpenAI: Generate listings
        OpenAI-->>ListingGenerator: Return generated listings
        ListingGenerator->>DallE: Generate images for listings
        DallE-->>ListingGenerator: Return image URLs
        ListingGenerator->>ListingGenerator: Save listings to pickle file
    end
    MainFunction->>CLIP: Load CLIP model and tokenizer
    MainFunction->>LanceDB: Connect and create/open table
    MainFunction->>CLIP: Generate embeddings for listings
    MainFunction->>LanceDB: Store listings with embeddings
    MainFunction->>CLIP: Encode user preferences
    MainFunction->>LanceDB: Search for matching listing
    LanceDB-->>MainFunction: Return matching listing
    MainFunction->>OpenAI: Enhance listing description
    OpenAI-->>MainFunction: Return enhanced description
    MainFunction-->>GradioInterface: Return image and description
    GradioInterface-->>User: Display recommendation
```

### User Interface
The application uses Gradio to create a user-friendly interface with the following inputs:

- Description (text input)
- State (dropdown)
- City (dropdown)
- Property Type (dropdown)
- Bedrooms (slider)
- Bathrooms (slider)
- Budget (slider)
-Amenities (checkbox group)

The output displays the recommended property image and its description.
### Data Storage

Listings are cached in a pickle file (listings.pickle) for quick access.

Images are stored in the images directory.

The LanceDB database is stored in the data/lancedb directory.

### AI Models

- OpenAI's GPT-3.5-turbo: Used for generating listing descriptions
- DALL-E: Used for generating property images
- CLIP: Used for generating embeddings and similarity search

## TODO Items

- Implement filtering in the find_listing() function using LanceDB's querying capabilities.
- Enhance the listing description using an LLM in the enhance_listing_description() function.
- Improve the embedding generation by incorporating more listing properties.

## Running the Application
To run the application, ensure all dependencies are installed and environment variables are set (particularly OpenAI API key). Then, simply run:
```bash
python HomeMatch.py
``` 
The Gradio interface will launch, allowing users to interact with the HomeMatch system.

