from app import app
from models import db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    email = input("Admin email: ")
    if User.query.filter_by(email=email).first():
        print("This email is already registered.")
    else:
        name = input("Admin name: ")
        password = input("Admin password: ")
        admin = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        print(f"Admin account created: {email}")