# ReadWise

**Find your next good read.** ReadWise is a book discovery app that helps you browse recommendations, explore book details, save titles to your shelf, and shape future suggestions with your feedback.

## Features

- **Browse books** by popularity, recency, and other discovery sections.
- **Get personalized recommendations** based on book ratings and your likes.
- **Give feedback** on books: Like, Not interested, or Already read.
- **Keep a reading history** with a card for each book marked as already read.
- **Save books to your shelf** for later.
- **Keep feedback private:** it is stored locally on your device and is not uploaded or treated as a Book-Crossing rating.
- **Filter out books** you have marked as not interested or already read.

## How recommendations use feedback

Likes can help guide future recommendations toward similar books. Books marked **Not interested** or **Already read** are excluded from recommendation results. Feedback is separate from the source ratings dataset and does not change its ratings.

## Tech stack

- Python
- Streamlit
- SQLite for local feedback storage
- Book-Crossing data for book and rating information

## Getting started

### Requirements

- Python 3.10 or newer
- Git

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

pip install -r requirements.txt
