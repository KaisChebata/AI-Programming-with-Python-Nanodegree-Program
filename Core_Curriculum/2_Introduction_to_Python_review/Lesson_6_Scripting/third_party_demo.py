from datetime import datetime
import pytz

utc = pytz.utc
algst = pytz.timezone('Africa/Algiers')

utc_now = datetime.now(tz=utc)
algst_now = utc_now.astimezone(algst)

print(f'UTC: {utc_now}')
print(f'Algeria Time: {algst_now}')
