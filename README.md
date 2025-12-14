# Hotel FAQ Chatbot using Multinomial Logical Regression

### Overview
- This project implements a machine-learning-based FAQ chatbot for an imaginary hotel named Astra Imperium Hotel, located in Kuala Lumpur.
- The chatbot operates in the hotel websites to answer users' queries
- The chatbot classifies user queries into predefined intents using Multinomial Logistic Regression, enabling fast and accurate responses to common hotel-related questions.

### Software Requirements
- Python 3.12.8

### Setup Instructions
1. Download or clone this project
2. Go to terminal in the project directory or virtual enviroment
3. Install dependencies: <code>pip install -r requirements.txt</code>
4. Run training.ipynb to train the model on the data in the "data" folder and save the trained artifacts in the "joblib" folder.
5. Run the <a href="">streamlit app</a>, keep in mind it loads pretrained joblib file stored in this repository

### Customizing Responses
- All chatbot responses are stored in the response folder:
  - response.json — maps intents to default chatbot replies
  - data.json — contains the dataset used for training
- To modify or add responses:
  1. Open the JSON file in any text editor.
  2. Edit existing responses or add new entries following the current structure.
  3. Save the file and retrain the model (if training data was updated) using training.ipynb.
- Changes in response.json will immediately reflect in the chatbot without retraining.

### List of all 35 intents
- add_night
- book_hotel
- book_parking_space
- bring_pets
- cancel_hotel_reservation
- cancellation_fees
- change_hotel_reservation
- check_child_policy
- check_functions
- check_hotel_facilities
- check_hotel_offers
- check_hotel_prices
- check_hotel_reservation
- check_in
- check_lost_item
- check_menu
- check_nearby_attractions
- check_out
- check_payment_methods
- check_room_availability
- check_room_type
- check_smoking_policy
- customer_service
- file_complaint
- get_refund
- goodbye
- greeting
- host_event
- human_agent
- invoices
- leave_review
- redeem_points
- search_hotel
- shuttle_service
- store_luggage
