import sys
import traceback

try:
    from app import app
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    with app.test_client() as c:
        rv = c.post('/login', data={'username': 'admin', 'password': 'a'}, follow_redirects=True)
        print('STATUS:', rv.status_code)
        body = rv.get_data(as_text=True)
        print('BODY START')
        print(body[:8000])
        print('BODY END')
except Exception:
    traceback.print_exc()
    sys.exit(1)
