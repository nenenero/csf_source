from flask import Flask, request, jsonify, render_template
import json
import pymysql

app = Flask(__name__)

#ps: 已换为本地代码
host = 'localhost'
user = 'root'
password = 'Wyc123456'
database = 'flightnus'

def get_db_connection():
    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/')
def index():
    return render_template('index19.html')

@app.route('/get_traffic/', methods=['POST'])
def get_traffic():
    data = request.json
    month = data.get('month')
    day = data.get('day')
    hour = data.get('hour')
    minute = data.get('minute')

    if month == 7:
        day += 30

    timeperiod = hour * 60 + minute
    timeperiod = max(5, min(1430, timeperiod))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = '''
            SELECT Flight_nums FROM airtraffic
            WHERE Day = %s AND TimePeriod <= %s AND TimePeriod >= %s
        '''
        cursor.execute(query, (day, timeperiod, timeperiod - 15))
        result = cursor.fetchone()
        conn.close()

        if result:
            return jsonify({"traffic_volume": result['Flight_nums']})
        else:
            return jsonify({"error": "No data found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_flight_data/', methods=['GET'])
def get_flight_data():
    try:
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        day_str = f'{day:02d}'
        json_path = f'./json/06{day_str}{hour}.json'

        with open(json_path, 'r') as f:
            flight_data = json.load(f)
        return jsonify(flight_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True)
