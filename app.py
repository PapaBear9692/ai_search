from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest

from search import run_pipeline

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/search")
def api_search():
    try:
        data = request.get_json(force=True)
        query = (data.get("query") or "").strip()
        if not query:
            raise BadRequest("Query is required.")

        user_query = "Find the contact information of " + query

        # No client control; fixed internally
        result = run_pipeline(user_query, max_candidates=20)
        return jsonify({"ok": True, "vendors": result.get("vendors", [])})

    except BadRequest as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
