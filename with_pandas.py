import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:Code%407338677189@localhost:3306/demo", echo = True)

df = pd.read_sql("SELECT * FROM people", con = engine)

print(df)

#new_data = pd.DataFrame({'name':['AK','SFQ'], 'age':[23,22]})
#new_data.to_sql('people', con=engine, if_exists='append',index = False)