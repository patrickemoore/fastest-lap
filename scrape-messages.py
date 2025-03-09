import requests
import json

# Replace this with your bot's token
TOKEN = 'YOUR_BOT_TOKEN_HERE'
# Replace this with your target channel ID
CHANNEL_ID = 1299055570904547348
# Discord API base URL
BASE_URL = 'https://discord.com/api/v10'

headers = {
    'Authorization': f'Bot {TOKEN}',
    'Content-Type': 'application/json'
}

def fetch_messages(channel_id):
    url = f"{BASE_URL}/channels/{channel_id}/messages"
    params = {
        'limit': 100
    }
    messages = []
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        batch = response.json()
        if not batch:
            break

        messages.extend(batch)
        params['before'] = batch[-1]['id']

    return messages

def main():
    messages = fetch_messages(CHANNEL_ID)
    for message in messages:
        print(f"{message['author']['username']}: {message['content']}")
    print(f'Total messages fetched: {len(messages)}')

if __name__ == "__main__":
    main()
