# Based News Reader

This project is a fully automated news digest web app. It fetches headlines from Google News, uses Google's Gemini AI to intelligently select and rank them based on user preferences, and serves the content from a live database.

The entire process is hosted on Railway, with a scheduled worker that updates the news hourly.

View the demo: **[news.blakerayvid.com](https://news.blakerayvid.com)**

---

## Key Features & How It Works

*   **Configurable via Google Sheets:** Easily manage topics, keywords, banned terms, and application parameters by copying and editing a [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing).
*   **Intelligent Curation with Gemini:** A carefully engineered prompt instructs the Gemini AI to perform aggressive cross-topic deduplication, filter out low-quality content, and prioritize headlines based on user-defined weights.
*   **Dynamic Web Application:**
    *   The backend is a lightweight Flask web server that queries a PostgreSQL database.
    *   Server-side rendering ensures fast load times, while historical data is lazy-loaded on demand.
*   **Automated Content Updates:**
    *   `digest.py`: Runs hourly to fetch and curate fresh news.
*   **Interactive Frontend:**
    *   Pure HTML, CSS, and vanilla JavaScript (dependency-free).
    *   Includes swipeable carousels for browsing past digests and market forecasts.
    *   **Chronological Topic Drill-down:** Click on any topic heading to view chronological updates filtered specifically for that topic, paginated day-by-day across the 10 most recent updates with navigation arrows, circular step indicators, and localized date subheaders.

## Tech Stack

*   **Backend:** Python 3, Flask
*   **AI & ML:** 
    *   Google Gemini API (`gemini-2.5-flash-lite`)
    *   `pandas`, `numpy`
*   **Database:** PostgreSQL
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript, Jinja2, `Chart.js`
*   **Hosting & Automation:** Railway (Web Service + Cron Jobs)

---

## Directory Structure

```plaintext
based-news/
├── digest.py                   # Worker script for fetching/curating news
├── Procfile                    # Defines processes for hosting
├── README.md                   # This file
├── requirements.txt            # Python dependencies for the worker
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

### 1. General Setup
1. **Clone** your forked repository:
   ```bash
   git clone <your-repository-url>
   cd based-news
   ```
2. **Create a `.env` file** in the project root containing your `GEMINI_API_KEY` and the `DATABASE_URL` from your Railway project.

---

### 2. Run the Background Worker (Root Context)
The background curation scripts run from the project root.
1. Create and activate a Python virtual environment in the project root:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the worker dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the news digest worker script:
   ```bash
   python digest.py
   ```

---

### 3. Run the Web Server (Web Context)
The Flask web interface is managed within the `web/` subdirectory and uses its own virtual environment.
1. **Navigate** into the web directory:
   ```bash
   cd web
   ```
2. Create and activate a separate virtual environment inside the `web/` directory:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the web service dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy or link your `.env` file from the project root into this `web/` folder so the Flask server can access your database configurations.
5. Run the web server:
   ```bash
   flask run
   ```
6. Open your browser and navigate to `http://localhost:5000`.

---

<br>

![](images/example.png)

---
<p align="center">&copy; 2026 <a href="https://blakerayvid.com">Blake Rayvid</a>. All rights reserved.</p>