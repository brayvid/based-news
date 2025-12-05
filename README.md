# Based News Reader

This project is a fully automated news digest and market intelligence web app. It fetches headlines from Google News, uses Google's Gemini AI to intelligently select and rank them based on user preferences, and serves the content from a live database. Additionally, it features a Machine Learning engine that analyzes news sentiment to forecast stock market trends.

The entire process is hosted on Railway, with scheduled worker processes that update the news and generate market predictions automatically.

View the demo: **[news.blakerayvid.com](https://news.blakerayvid.com)**

---

## Key Features & How It Works

*   **Configurable via Google Sheets:** Easily manage topics, keywords, banned terms, and application parameters by copying and editing a [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing).
*   **Intelligent Curation with Gemini:** A carefully engineered prompt instructs the Gemini AI to perform aggressive cross-topic deduplication, filter out low-quality content, and prioritize headlines based on user-defined weights.
*   **AI Market Forecast:**
    *   **The Engine:** A Random Forest Classifier trains on historical news headlines aligned with market data to predict the one-week directional trend (Up/Down) of the S&P 500 (SPY).
    *   **Transparency:** The model identifies specific "evidence" headlines that contributed most significantly to the prediction.
    *   **Visualization:** An interactive dashboard displays the forecast direction, confidence score, and a synchronized historical price chart.
*   **Dynamic Web Application:**
    *   The backend is a lightweight Flask web server that queries a PostgreSQL database.
    *   Server-side rendering ensures fast load times, while historical data is lazy-loaded on demand.
*   **Automated Content Updates:**
    *   `digest.py`: Runs hourly to fetch and curate fresh news.
    *   `engine.py`: Runs before market open to retrain the model and generate a prediction based on overnight news.
*   **Interactive Frontend:**
    *   Pure HTML, CSS, and vanilla JavaScript (dependency-free).
    *   Includes swipeable carousels for browsing past digests and market forecasts.
    *   Forecasting charts are rendered using `Chart.js`.

## Tech Stack

*   **Backend:** Python 3, Flask
*   **AI & ML:** 
    *   Google Gemini API (`gemini-2.5-flash-lite`)
    *   `scikit-learn` (Random Forest Classifier)
    *   `pandas`, `numpy`
*   **Market Data:** `yfinance`, `pandas_market_calendars`
*   **Database:** PostgreSQL
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript, Jinja2, `Chart.js`
*   **Hosting & Automation:** Railway (Web Service + Cron Jobs)

---

## Directory Structure

```plaintext
based-news/
├── digest.py                   # Worker script for fetching/curating news
├── forecast/
│   └── engine.py               # ML script for training and market prediction
├── Procfile                    # Defines processes for hosting
├── README.md                   # This file
├── requirements.txt            # Python dependencies for the workers
└── web/
    ├── app.py                  # Flask web server
    ├── railway.json            # Railway deployment configuration
    ├── requirements.txt        # Python dependencies for the web server
    ├── static/                 
    │   └── favicon.ico         
    └── templates/
        ├── index.html          # Template for the news digest
        └── forecast.html       # Template for the market forecast
```

---

## Local Development

1.  **Clone** your forked repository.
2.  **Create and activate** a Python virtual environment.
3.  **Install dependencies:** `pip install -r requirements.txt` and `pip install -r web/requirements.txt`.
4.  **Create a `.env` file** with your `GEMINI_API_KEY` and the `DATABASE_URL` from your Railway project.
5.  **Run the news digest worker:** `python digest.py`.
6.  **Run the forecast engine:** `python forecast/engine.py`.
7.  **Run the web server** to view the site: `python web/app.py` and navigate to `http://localhost:5000`.

---

<br>

![](images/example.png)