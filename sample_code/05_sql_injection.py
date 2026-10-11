# Sample 5: SQL Injection via f-strings and string concatenation

import sqlite3

def get_user_by_id(cursor, user_id):
    # CRITICAL: Vulnerable SQL query via f-string
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()


def delete_user_by_email(cursor, email):
    # CRITICAL: Vulnerable SQL query via string concatenation
    query = "DELETE FROM users WHERE email = '" + email + "'"
    cursor.execute(query)
