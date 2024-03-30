from flask import Flask, render_template, redirect, request, url_for, jsonify
import requests
import json
import csv



# Strava API credentials
CLIENT_ID = '122835'
CLIENT_SECRET = '1bed8978a9dd9dda86e30145a669781e087c76af'
REDIRECT_URI = 'http://127.0.0.1:5000/callback'  # Update with your actual Redirect URI

app = Flask(__name__)


def get_athlete_activities(access_token, page=1, per_page=50):
    # Example: Get the logged-in athlete's activities
    activities_url = 'https://www.strava.com/api/v3/athlete/activities'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'page': page, 'per_page': per_page}

    try:
        activities_response = requests.get(activities_url, headers=headers, params=params)
        activities_response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        activities_data = activities_response.json()
        return activities_data

    except requests.exceptions.RequestException as e:
        print('error: Request failed:' + str(e))
        print("failed")
        return {'error': f'Request failed: {e}'}




def get_athlete_stats(access_token, athlete_id, csv_file_path):
    # Update the athlete ID based on your requirements
    stats_url = f'https://www.strava.com/api/v3/athletes/{athlete_id}/stats'
    headers = {'Authorization': f'Bearer {access_token}'}

    response = requests.get(stats_url, headers=headers)
    selected_keys = response.json().keys()
    with open(csv_file_path + "_stats.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=selected_keys)
        writer.writeheader()
        writer.writerow(response.json())
    return response.json()


def get_json_data(access_token):
    all_data = []
    for i in range(1, 25):
        json_data = jsonify(get_athlete_activities(access_token, i, 50)).get_data(as_text=True)
        all_data.extend(json.loads(json_data))
    return all_data

def write_athlete_data_to_csv(data, csv_file_path, selected_keys):
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=selected_keys)
        writer.writeheader()
        for row in data:
            # Extract only the selected keys from each JSON object
            selected_data = {key: row.get(key, '') for key in selected_keys}
            writer.writerow(selected_data)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login")
def login():
    strava_authorize_url = (
        'https://www.strava.com/oauth/authorize?client_id={}&redirect_uri={}&response_type=code&scope=read_all,activity:read_all'
    ).format(CLIENT_ID, REDIRECT_URI)
    return redirect(strava_authorize_url)

@app.route("/callback")
def callback():
    code = request.args.get('code')
    token_url = 'https://www.strava.com/oauth/token'
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI
    }
    response = requests.post(token_url, data=payload)
    response_data = response.json()
    print("Strava API Response:", response_data)
    access_token = response.json().get('access_token')

    # get athlete's name
    firstName = str(response_data.get('athlete')["firstname"])
    lastName = str(response_data.get('athlete')["lastname"])
    csv_file_path = firstName + "_" + lastName + "_activities.csv"

    # get athlete's overall stats
    print("athlete id: " + str(response_data.get('athlete')["firstname"]))
    get_athlete_stats(access_token, response_data.get('athlete')["id"], firstName + "_" + lastName)

    # get athlete's activites and write to a csv file
    selected_keys = ['name', 'type', 'start_date', 'athlete_count', 'average_cadence', 'average_heartrate',
                     'average_speed', 'distance', 'elapsed_time', "has_heartrate", 'max_heartrate', 'max_speed',
                     'moving_time', 'sport_type', 'total_elevation_gain', 'kudos_count', 'achievement_count',
                     'pr_count']

    data = get_json_data(access_token)
    write_athlete_data_to_csv(data, csv_file_path, selected_keys)
    return "Success"

if __name__ == '__main__':
    app.run()