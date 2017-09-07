# This script generates the scoring and schema files
# necessary to opearaitonalize the Iris Classification sample
# Init and run functions
from azure.ml.api.schema.dataTypes import DataTypes
from azure.ml.api.schema.sampleDefinition import SampleDefinition
from azure.ml.api.realtime.services import prepare
import pandas

# Prepare the web service definition by authoring
# init() and run() functions. Test the fucntions
# before deploying the web service.
def init():
    from sklearn.externals import joblib

    # load the model file
    global model
    model = joblib.load('model.pkl')

def run(input_df):
    import json
    pred = model.predict(input_df)
    return json.dumps(str(pred[0]))

df = pandas.DataFrame(data=[[3.0, 3.6, 1.3, 0.25]], columns=['sepal length', 'sepal width','petal length','petal width'])
df.dtypes
df

init()
input1 = pandas.DataFrame([[3.0, 3.6, 1.3, 0.25]])
run(input1)

inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}
# The prepare statement writes the scoring file (main.py) and
# the scchema file (service_schema.json) the the output folder.
prepare(run_func=run, init_func=init, input_types=inputs, )