# **Installation**

## **1. Create a `.env` File for the Backend**

Follow these steps to set up and run the project on your local machine.

Before running the backend, create a `.env` file inside the `backend/` folder with the following content:

```env
PORT=5000
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_secret_jwt_key
```

 Replace `your_mongodb_connection_string` and `your_secret_jwt_key` with your actual values.

---

## **2. Install Dependencies & Start the Backend**
Navigate to the `backend/` folder and install dependencies:

```sh

cd backend
npm install

```

Then, start the backend using **nodemon**:

```sh

nodemon index.js

```

## **3️⃣ Install Dependencies & Start the Frontend**
In a new terminal window, navigate to the `frontend/` folder and install dependencies:

```sh

cd frontend
npm install

```

Start the frontend with:

```sh

npm run dev

```

---

Now your backend runs on **http://localhost:5000** (or the port you specified), and the frontend runs on available port.

---