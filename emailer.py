"""
DEPRECATED / UNUSED -- kept only because this session's tools can't delete
files outright. Nothing imports this module anymore.

The "email me this poster" flow was removed and replaced with an
Instagram-share prompt on the reveal screen (see app.py's ARIES_INSTAGRAM_URL
/ ARIES_INSTAGRAM_HANDLE and index.html's ig-share-box). GMAIL_ADDRESS /
GMAIL_APP_PASSWORD / EMAIL_FROM_NAME in .env are no longer read by anything.

Safe to delete this file entirely next time you're on a machine where you
can remove it (`rm emailer.py` or delete via your file browser) -- nothing
in the app will break, since app.py no longer imports it.
"""
