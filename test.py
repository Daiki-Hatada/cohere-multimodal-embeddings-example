import base64
import os

import cohere
import numpy as np
from dotenv import load_dotenv


def main():
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    ### STEP 1: Embed the images
    embeddings_list = []

    image_paths = ["./data/canadian-passport.jpg", "./data/japanese-passport.jpg", "./data/smith-zairyu.jpg"]
    for image_path in image_paths:
        _, file_extension = os.path.splitext(image_path)
        file_type = file_extension[1:]
        with open(image_path, "rb") as f:
            enc_img = base64.b64encode(f.read()).decode("utf-8")
            enc_img = f"data:image/{file_type};base64,{enc_img}"
            embeddings = co.embed(
                images=[enc_img],
                model="embed-multilingual-v3.0",
                input_type="image",
                embedding_types=["float"],
            ).embeddings.float
            embeddings_list.append(*embeddings)

    ### STEP 2: Embed the image as query
    image_path = "./data/elizabeth-zairyu.png"
    _, file_extension = os.path.splitext(image_path)
    file_type = file_extension[1:]
    with open(image_path, "rb") as f:
        enc_img = base64.b64encode(f.read()).decode("utf-8")
        enc_img = f"data:image/{file_type};base64,{enc_img}"
        query_emb = co.embed(
            images=[enc_img],
            model="embed-multilingual-v3.0",
            input_type="image",
            embedding_types=["float"],
        ).embeddings.float

    ### STEP 3: Return the most similar images by dot product
    scores = np.dot(query_emb, np.transpose(embeddings_list))[0]
    print(scores)

    top_n = 3
    top_doc_idxs = np.argsort(-scores)[:top_n]

    for idx, docs_idx in enumerate(top_doc_idxs):
        print(f"Rank: {idx + 1}")
        print(docs_idx)


if __name__ == "__main__":
    load_dotenv()
    main()
