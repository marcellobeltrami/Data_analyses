import streamlit as st


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Salaries analysis", "Page 2", "Page 3", "Page 4"],
)

# Import pages dynamically based on selection
if page == "Home":
    st.title("Home")
    
    st.write("""
    Hello, 

    Welcome. Here you will be able to see the results of the analyses I conducted using various kaggle datasets. 
    Before we start here is a little about myself. My name is Marcello Beltrami and I am currently working as a bionformatician at the University of York. 
    I have a keen interest in data analysis and machine learning, and I am excited to share my work. 
    I am very open to feedback and suggestions, so please don't hesitate to reach out if you have any questions or suggestions. 
    I truely enjoy working with others and when possible mentor them a little.
    In my freetime I enjoy a little climbing and parkour. I am a huge fan of travelling and really look forward to connect with people from all around the world.
    """)
elif page == "Salaries analysis":
    import salaries_analysis.dashboard_page