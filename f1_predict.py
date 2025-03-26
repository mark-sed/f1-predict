import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def to_milliseconds_dot(t):
    if t == '\\N':  # Keep '\\N' as is
        return '\\N'
    mm, rest = t.split(':')
    ss, ms = rest.split('.')
    return int(mm) * 60000 + int(ss) * 1000 + int(ms)

races_data = pd.read_csv("archive/races.csv", header=0)
races_data['time_hour'] = (
    races_data['time'].str.split(':').apply(lambda x: x[0])
)

sprints_data = pd.read_csv("archive/sprint_results.csv", header=0)
sprints_data.rename(columns={"resultId": "sp_resultId", "driverId": "sp_driverId", "constructorId": "sp_constructorId", "number": "sp_number", "grid": "sp_grid", "position": "sp_position", "positionText": "sp_positionText", "positionOrder": "sp_positionOrder", "points": "sp_points",
    "laps": "sp_laps","time": "sp_time","milliseconds": "sp_milliseconds","fastestLap": "sp_fastestLap","fastestLapTime": "sp_fastestLapTime","statusId": "sp_statusId"}, inplace=True)
sprints_data['sp_fastest_lap_ms'] = sprints_data['sp_fastestLapTime'].apply(to_milliseconds_dot)

col_names_results = ["resultId","raceId","driverId","constructorId","number","grid","position","positionText","positionOrder","points","laps","time","milliseconds","fastestLap","rank","fastestLapTime","fastestLapSpeed", "statusId"]
results_data = pd.read_csv("archive/results.csv", header=0, names=col_names_results)
# Remove all unfinished, but maybe TODO: Replace with inf?
results_data = results_data.loc[results_data['milliseconds'] != '\\N']

results_data = pd.merge(results_data, races_data, on="raceId")
results_data = pd.merge(results_data, sprints_data, on="raceId")

results_data = results_data[results_data["year"] >= 2010]
results_data = results_data.apply(pd.to_numeric, errors='coerce')

feature_cols = ["driverId","constructorId","laps","year", "round", "circuitId", "time_hour", "sp_position", "sp_milliseconds"] #, "sp_fastest_lap_ms", "sp_fastestLap"]
x = results_data[feature_cols]
y = results_data.position

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # 80% training and 20% test

clf = HistGradientBoostingClassifier() # or DecisionTreeClassifier

# Train Decision Tree Classifer
clf = clf.fit(x, y)

#Predict the response for test dataset
#print("XTEST")
#print(x_test.head())
#y_pred = clf.predict(x_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

drivers_data = pd.read_csv("archive/drivers.csv", header=0)
constructors_data = pd.read_csv("archive/constructors.csv", header=0)
circuits_data = pd.read_csv("archive/circuits.csv", header=0)

driver = input("Driver code (e.g. GAS): ")
lookup_dr = drivers_data[drivers_data['code'] == driver]['driverId']
if not lookup_dr.empty:
    driverId = lookup_dr.iloc[0]
else:
    print("Error: Unknown name")
    exit(1)

constr = input("Constructor code (e.g. alpine): ")
lookup_cr = constructors_data[constructors_data['constructorRef'] == constr]['constructorId']
if not lookup_cr.empty:
    constructorId = lookup_cr.iloc[0]
else:
    print("Error: Unknown constructor name")
    exit(1)

circuit = input("Circuit ID: ")
circuitId = int(circuit)
#circuitId = 22

round = input("What round is this: ")
#round = 1
laps = input("How many laps are there: ")
#laps = 53
time = input("At what local HOUR does the race start: ")
#time = 14

sprint_position_txt = input("What was his sprint position (N/A otherwise): ")
try:
    sprint_pos = int(sprint_position_txt)
except Exception:
    sprint_pos = float("nan")

sprint_time_ms_txt = input("What was his sprint time in MM:SS.ms (N/A otherwise): ")
try:
    sprint_time_ms = to_milliseconds_dot(sprint_time_ms_txt)
except Exception:
    sprint_time_ms = float("nan")

name = drivers_data[drivers_data["driverId"] == driverId]['surname']
const_name = constructors_data[constructors_data["constructorId"] == constructorId]['name'].iloc[0]
circuit_name = circuits_data[circuits_data["circuitId"] == circuitId]["name"].iloc[0]
print(f"Looking up: {name.iloc[0]} in team {const_name} in round {round} at {circuit_name} ({laps} laps)")
print(f"Sprint position: {sprint_pos} @ {sprint_time_ms} ms")

pred_frame = pd.DataFrame({
    "driverId": [driverId],
    "constructorId": [constructorId],
    "laps": [int(laps)],
    "year": [2024],
    "round": [int(round)],
    "circuitId": [circuitId],
    "time_hour": [time],
    "sp_position": [sprint_pos],
    "sp_milliseconds": [sprint_time_ms],
})

pred_result = clf.predict(pred_frame)
print(pred_result)