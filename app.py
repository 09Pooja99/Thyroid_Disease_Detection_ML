from flask import Flask

app= Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    return "Thyroid Disease Detection Using Machine Learing"

if __name__ == "__main__":
    app.run(debug= True)
