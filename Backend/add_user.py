from backend import app, db, User

with app.app_context():
	new_user = User(username = "Benas", phone_number="+447399963248")
	new_user.set_password("secret123")
	db.session.add(new_user)
	db.session.commit()
	print("User Benas added successfully!")
