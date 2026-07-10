# Based News Reader

This project is a fully automated news digest web app. It fetches headlines from Google News, uses Google's Gemini AI to intelligently select and rank them based on user preferences, and serves the content from a live database.

The entire process is hosted on Railway, with a scheduled worker that updates the news hourly.

View the demo: **[news.blakerayvid.com](https://news.blakerayvid.com)**

---

## Key Features

*   **Configurable via Google Sheets:** Easily manage topics, keywords, banned terms, and application parameters by copying and editing a [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing).
*   **Intelligent Curation with Gemini:** A carefully engineered prompt instructs the Gemini AI to perform aggressive cross-topic deduplication, filter out low-quality content, and prioritize headlines based on user-defined weights.
*   **Dynamic Web App:**
    *   The backend is a lightweight Flask web server that queries a PostgreSQL database.
    *   Server-side rendering ensures fast load times, while historical data is lazy-loaded on demand.
*   **Automated Updates:**
    *   `digest/digest.py`: Runs hourly to fetch and curate fresh news.
*   **Interactive Frontend:**
    *   Pure HTML, CSS, and vanilla JavaScript (dependency-free).
    *   Includes swipeable carousels for browsing past digests.
    *   **Single-topic mode:** Click on any topic heading to view chronological updates filtered specifically for that topic. This drills down into a single, scrollable page displaying the 10 most recent days of headlines, grouped chronologically by day headers with the newest headlines at the top.
    *   **Accessible Navigation:** Supports keyboard controls (`ArrowLeft` / `ArrowRight` to transition digests, `Escape` to close topic views), pointer hovers, and clean hierarchical heading layouts.

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
├── README.md                   # This file
├── web/
    ├── app.py                  # Flask web server
    ├── railway.json            # Railway deployment configuration
    ├── requirements.txt        # Python dependencies for the web server
    ├── static/                 
    │   └── favicon.ico         
    └── templates/
        └── index.html          # Template for the news digest
└── digest/
    ├── digest.py               # Worker script for fetching/curating news
    ├── Procfile                # Defines processes for hosting
    └── requirements.txt        # Python dependencies for the digest worker    
```

---

## Local Development

### 1. General Setup
1. **Clone** your forked repository:
   ```bash
   git clone <your-repository-url>
   cd based-news
   ```

---

### 2. Run the Background Worker (Digest Context)
The background curation script and its dependencies are managed entirely within the `digest/` subdirectory.
1. **Navigate** into the digest directory:
   ```bash
   cd digest
   ```
2. Create and activate a virtual environment inside the `digest/` directory:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: source .venv/Scripts/activate
   ```
3. Install the background worker dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file** inside the `digest/` folder with your Google Gemini API Key and Database connection string (needed to write curated articles to the database):
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```
5. Run the news digest worker script:
   ```bash
   python digest.py
   ```

---

### 3. Run the Web Server (Web Context)
The Flask web interface is managed within the `web/` subdirectory and uses its own virtual environment.
1. **Navigate** into the web directory (from the project root):
   ```bash
   cd ../web
   ```
2. Create and activate a separate virtual environment inside the `web/` directory:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: source .venv/Scripts/activate
   ```
3. Install the web service dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file** inside the `web/` folder with your Database connection string (needed for the Flask server to read updates):
   ```env
   DATABASE_URL=your_postgresql_database_url
   ```
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