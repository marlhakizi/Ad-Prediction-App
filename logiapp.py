from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
app = Flask(__name__)
clf = joblib.load('logistic.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
            #rr=request.json
            #tr=pd.read_json(rr)
            train3 = pd.DataFrame(request.get_json(force=True)) #getting first input
            #train3=pd.read_json(rr)
            #item_log=json2_.merge(json1_,how='left',on='item_id') #merging items and log
            #train3=json_.merge(query,how='left',on='user_id') # merging items,log anf train
            colcol=['impression_id', 'impression_time','user_id', 'app_code', 'os_version','is_4G']
            alltrain=train3.reindex(columns=colcol, fill_value=0).drop_duplicates()

            cat_agg=['count','nunique']
            num_agg=['min','mean','max','sum']
            agg_col={'server_time':'nunique',
            'session_id':'nunique','item_price':'mean',
            'category_3':['nunique','mean'], 'product_type':['nunique','mean']
            }

            for k in train3.columns:
                if k.startswith('category_1') or k.startswith('category_2'):
                    agg_col[k]=['sum','mean']
                elif k.startswith('server'):
                    agg_col[k]=cat_agg
                elif k.startswith('cumcount'):
                    agg_col[k]=num_agg


            untrain=train3.groupby('impression_id').agg(agg_col)
            untrain.columns=['J_' + '_'.join(col).strip() for col in untrain.columns.values]
            on=untrain.reset_index()
            allallu=on.merge(alltrain,how='left',on='impression_id')


            allallu.loc[:,'impression_time']=pd.to_datetime(allallu['impression_time'])
            allallu['Hour']=allallu.loc[:,'impression_time'].dt.hour
            allallu['Day']=allallu.loc[:,'impression_time'].dt.day


            allallu['newHour']=pd.cut(allallu.Hour,bins=[0,6,12,17,23],labels=['Early','Morning','Afternoon','Night'],include_lowest=True)


            hou=pd.get_dummies(allallu.newHour)
            ensemble=pd.concat([allallu,hou],axis=1)


            rty=['impression_id', 'item_id','impression_time','user_id','Hour','os_version',
            'server_time_y', 'impression_id_y','newHour']

            rtrt=[i for i in ensemble.columns if i not in rty]
            ensemble1=ensemble[rtrt].fillna(0)



            prediction = list(clf.predict(ensemble1))
            prediction_str=[str(i) for i in prediction]

            return jsonify({'is_click': prediction_str})
if __name__ == '__main__':
    #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')
    app.run(debug=True)
