from flask import Flask, request, jsonify
from PipeLined import get_ratings  # Ensure PipeLined.py is in the same directory

app = Flask(__name__)

@app.route('/evaluate-comment', methods=['POST'])
def evaluate_comment():
    data = request.get_json()
    comment_text = data.get('comment')
    hate_rating, spam_rating = get_ratings(comment_text)
    print('Result:', jsonify({'hate_rating': hate_rating, 'spam_rating': spam_rating}))
    return jsonify({'hate_rating': hate_rating, 'spam_rating': spam_rating})

if __name__ == '__main__':
    app.run(debug=True)
