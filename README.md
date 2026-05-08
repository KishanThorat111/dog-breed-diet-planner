# Dog Breed Diet Planner

A production-ready Flask web application for identifying dog breeds from images and providing personalized diet plans.

## Features

- User registration and authentication
- Admin panel for user management
- Image upload and breed prediction using TensorFlow
- Diet recommendations based on breed size
- Report generation and download
- Responsive UI with Bootstrap

## Tech Stack

- **Backend**: Flask, SQLAlchemy, TensorFlow
- **Frontend**: Jinja2 templates, Bootstrap, CSS
- **Database**: SQLite (configurable to PostgreSQL)
- **Deployment**: Vercel (serverless)

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd dog-breed-diet-planner
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

5. Run the application:
   ```bash
   python run.py
   ```

## Deployment to Vercel

1. Push code to GitHub.

2. Connect Vercel to your GitHub repo.

3. Vercel will automatically detect `vercel.json` and deploy.

Note: For production, consider using a cloud database and storage service instead of local files.

## Project Structure

```
├── app/
│   ├── __init__.py          # App factory
│   ├── config.py            # Configuration
│   ├── diet_data.py         # Diet plans data
│   ├── models/
│   │   └── user.py          # User model
│   ├── routes/
│   │   ├── auth.py          # Auth routes
│   │   ├── main.py          # Main app routes
│   │   └── admin.py         # Admin routes
│   └── utils/
│       └── ml_inference.py  # ML prediction logic
├── static/                  # Static files (CSS, images)
├── templates/               # Jinja2 templates
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── run.py                   # App entry point
├── vercel.json              # Vercel deployment config
└── README.md                # This file
```

## Security Notes

- Passwords are hashed using Werkzeug.
- CSRF protection enabled.
- Admin credentials stored in environment variables.
- File uploads sanitized.

## Future Improvements

- Migrate to cloud storage (AWS S3).
- Use a dedicated ML API for predictions.
- Add user profiles and history.
- Implement proper logging and monitoring.