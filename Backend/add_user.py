from backend import app, db, Admin

with app.app_context():
	new_admin = Admin(username = "Benas", phone_number="+447399963248")
	new_admin.set_password("secret123")
	db.session.add(new_admin)
	db.session.commit()
	print("Admin Benas added successfully!")
