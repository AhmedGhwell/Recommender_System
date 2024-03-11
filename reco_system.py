import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import scikit-learn 
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(data):
    # Drop unnecessary columns
    data.drop(['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status'], axis=1, inplace=True)
    return data

def get_recommendations(data, product_id_input):
    # Create a user-item matrix
    user_item_matrix = data.pivot(index='Product_ID', columns='User_ID', values='Purchase').fillna(0)

    # Calculate cosine similarity
    item_similarity = cosine_similarity(user_item_matrix)

    # Get the index of the input product
    product_index = user_item_matrix.index.get_loc(product_id_input)

    # Calculate cosine similarity for the input product
    similar_products_indices = np.argsort(item_similarity[product_index])[::-1]

    # Top-N similar products (excluding the input product itself)
    N = 5
    top_similar_products = user_item_matrix.index[similar_products_indices[1:N + 1]]

    return top_similar_products.tolist()

def main():
    # Streamlit app configuration
    st.set_page_config(page_title="Product Recommendation System", layout="wide")

    # Streamlit UI components
    st.title("Product Recommendation System")
    st.markdown("Upload a CSV file and enter a Product_ID to get product recommendations:")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Display uploaded data
    if uploaded_file is not None:
        st.subheader("Please go to my Github repo and download the dataset incase you do not have it visit this url: https://github.com/AhmedGhwell/Recommender_System")
        st.subheader("Uploaded Data:")
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # User input for Product_ID
        product_id_input = st.text_input("Enter Product_ID:", "P00069042")

        # Recommendation logic
        if st.button("Get Recommendations"):
            try:
                # Preprocess data
                data = preprocess_data(data)

                # Get recommendations
                recommendations = get_recommendations(data, product_id_input)

                # Display recommendations
                st.subheader("Recommended Products:")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
