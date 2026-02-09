from flask import Flask, render_template,request,jsonify
from RAG_core.main import incomming_query


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    print("User asked:", question)

    answer = incomming_query(question)

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True)
