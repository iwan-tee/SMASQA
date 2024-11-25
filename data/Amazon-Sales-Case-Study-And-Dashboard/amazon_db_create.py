import os
import sqlite3
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, 'Amazon.csv')

db_path = os.path.join(script_dir, 'amazon.db')

amazon_data = pd.read_csv(csv_path)

conn = sqlite3.connect(db_path)

amazon_data.to_sql('amazon', conn, if_exists='replace', index=False)

with conn:
    # time_of_day
    conn.execute("ALTER TABLE amazon ADD COLUMN time_of_day TEXT;")
    conn.execute("""
        UPDATE amazon
        SET time_of_day = CASE
            WHEN CAST(strftime('%H', time) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
            WHEN CAST(strftime('%H', time) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
            ELSE 'Evening'
        END;
    """)

    # day_name
    conn.execute("ALTER TABLE amazon ADD COLUMN day_name TEXT;")
    conn.execute("""
        UPDATE amazon
        SET day_name = CASE strftime('%w', date)
            WHEN '0' THEN 'Sunday'
            WHEN '1' THEN 'Monday'
            WHEN '2' THEN 'Tuesday'
            WHEN '3' THEN 'Wednesday'
            WHEN '4' THEN 'Thursday'
            WHEN '5' THEN 'Friday'
            WHEN '6' THEN 'Saturday'
        END;
    """)

    # Month
    conn.execute("ALTER TABLE amazon ADD COLUMN month_name TEXT;")
    conn.execute("""
        UPDATE amazon
        SET month_name = CASE strftime('%m', date)
            WHEN '01' THEN 'January'
            WHEN '02' THEN 'February'
            WHEN '03' THEN 'March'
            WHEN '04' THEN 'April'
            WHEN '05' THEN 'May'
            WHEN '06' THEN 'June'
            WHEN '07' THEN 'July'
            WHEN '08' THEN 'August'
            WHEN '09' THEN 'September'
            WHEN '10' THEN 'October'
            WHEN '11' THEN 'November'
            WHEN '12' THEN 'December'
        END;
    """)

with conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS amazon_sales (
            invoice_id TEXT PRIMARY KEY NOT NULL,
            branch TEXT NOT NULL,
            city TEXT NOT NULL,
            customer_type TEXT NOT NULL,
            gender TEXT NOT NULL,
            product_line TEXT NOT NULL,
            unit_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            vat REAL NOT NULL,
            total REAL NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            cogs REAL NOT NULL,
            gross_margin_percentage REAL NOT NULL,
            gross_income REAL NOT NULL,
            rating REAL NOT NULL,
            time_of_day TEXT NOT NULL,
            day_name TEXT NOT NULL,
            month_name TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO amazon_sales SELECT * FROM amazon;")

test_query = pd.read_sql("SELECT * FROM amazon_sales LIMIT 10;", conn)
print(test_query)

conn.close()

