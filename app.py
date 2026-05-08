# app.py — thin shim kept for backwards compatibility (e.g. Vercel's run.py entrypoint).
# All application logic lives in the app/ package.
# The legacy Excel-based user store has been removed; the app/ package uses SQLAlchemy.
from app import create_app

application = create_app()  # WSGI entrypoint name expected by some hosts

if __name__ == '__main__':
    application.run(debug=False)

@app.route('/')
def home():
    return redirect(url_for('login'))


# REGISTER
@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template('register.html', message="All fields required!")

        success = register_user(username, email, password)

        if not success:
            return render_template('register.html', message="Email already exists!")

        return redirect(url_for('login'))

    return render_template('register.html')


# USER LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('login.html', message="All fields required!")

        user = check_user(email, password)

        if user:
            session['user'] = str(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message="Invalid Credentials")

    return render_template('login.html')


# ADMIN LOGIN
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email == ADMIN_EMAIL and check_password_hash(ADMIN_PASSWORD, password):
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))

        return render_template('admin_login.html', message="Invalid Admin Credentials")

    return render_template('admin_login.html')


# LOGOUT
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/admin_logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))


# MAIN APP
@app.route('/index', methods=['GET', 'POST'])
def index():

    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':

        file = request.files['image']

        if file and allowed_file(file.filename):

            # UNIQUE filename
            filename = str(datetime.datetime.now().timestamp()).replace(".", "") + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            breed, confidence = predict_breed(filepath)
            size = get_size_category(breed)
            diet = diet_plans[size]

            youtube_link = f"https://www.youtube.com/results?search_query={breed}+dog+training"
            amazon_link = "https://www.amazon.in/s?k=" + diet["food"].replace(" ", "+")

            # SAFE SESSION STORAGE
            session['breed'] = str(breed)
            session['confidence'] = float(confidence)
            session['size'] = str(size)
            session['diet'] = {
                "food": str(diet["food"]),
                "meals": str(diet["meals"]),
                "extras": str(diet["extras"])
            }

            return render_template(
                'result.html',
                image=filepath,
                breed=breed,
                confidence=confidence,
                diet=diet,
                size=size,
                youtube_link=youtube_link,
                amazon_link=amazon_link,
                user=session['user']
            )

    return render_template('index.html', user=session['user'])


# DOWNLOAD REPORT
@app.route('/download_report')
def download_report():

    if 'user' not in session:
        return redirect(url_for('login'))

    diet = session.get('diet', {})

    report = f"""
Dog Report
---------------------
Breed: {session.get('breed')}
Confidence: {session.get('confidence')} %

Size: {session.get('size')}

Food: {diet.get('food', 'N/A')}
Meals: {diet.get('meals', 'N/A')}
Extras: {diet.get('extras', 'N/A')}

Date: {datetime.datetime.now()}
"""

    file_path = "report.txt"
    with open(file_path, "w") as f:
        f.write(report)

    return send_file(file_path, as_attachment=True)


# ADMIN DASHBOARD
@app.route('/admin')
def admin_dashboard():

    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if not os.path.exists(EXCEL_FILE):
        return render_template("admin.html", users=[])

    wb = openpyxl.load_workbook(EXCEL_FILE)
    sheet = wb.active

    users = []

    for i, row in enumerate(sheet.iter_rows(min_row=2, values_only=True)):
        users.append({
            "id": i,
            "username": row[0],
            "email": row[1],
            "password": row[2]
        })

    return render_template("admin.html", users=users)


# DELETE USER
@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):

    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    wb = openpyxl.load_workbook(EXCEL_FILE)
    sheet = wb.active

    sheet.delete_rows(user_id + 2)
    wb.save(EXCEL_FILE)

    return redirect(url_for('admin_dashboard'))


# EDIT USER
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):

    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    wb = openpyxl.load_workbook(EXCEL_FILE)
    sheet = wb.active
    row_no = user_id + 2

    if request.method == 'POST':
        sheet.cell(row=row_no, column=1).value = request.form['username']
        sheet.cell(row=row_no, column=2).value = request.form['email']
        sheet.cell(row=row_no, column=3).value = generate_password_hash(request.form['password'])

        wb.save(EXCEL_FILE)
        return redirect(url_for('admin_dashboard'))

    user = {
        "username": sheet.cell(row=row_no, column=1).value,
        "email": sheet.cell(row=row_no, column=2).value
    }

    return render_template("edit_user.html", user=user)


# EXTRA PAGES
@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', user=session['user'])


@app.route('/services')
def services():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('services.html', user=session['user'])


@app.route('/vets')
def vets():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('vets.html', user=session['user'])


# ==============================
if __name__ == '__main__':
    app.run(debug=True)
