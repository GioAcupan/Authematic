from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Load environment variables so the underlying modules can access API keys
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from run_pipeline import run_pipeline  # noqa: E402

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form.get('title', '')
        cutoff_year = int(request.form.get('cutoff_year', '0'))
        citation_style = request.form.get('citation_style', '')
        results = run_pipeline(title, cutoff_year, citation_style)
        return render_template('results.html', title=title, results=results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
