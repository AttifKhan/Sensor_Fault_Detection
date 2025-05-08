from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, insert, Float, ForeignKey

# for any file without connecting to a database
#engine = create_engine('sqlite:///mydatabase.db', echo = True)

engine = create_engine("mysql+pymysql://root:Code%407338677189@localhost:3306/demo", echo = True)


"""
conn = engine.connect()

# To write raw sql code we need to wrap it in text
conn.execute(text("CREATE TABLE IF NOT EXISTS people (name str, age int)"))

conn.commit()

from sqlalchemy.orm import Session

session = Session(engine)

session.execute(text('INSERT INTO people (name, age) VALUES ("Mile", 30)'))

session.commit()"""

meta = MetaData()

people = Table(
    "people",
    meta,
    Column('id', Integer, primary_key=True),
    Column('name', String(20), nullable= False),
    Column('age', Integer)
)

things = Table(
    "things",
    meta,
    Column('id',Integer, primary_key=True),
    Column('description', String(50), nullable = False),
    Column('value',Float),
    Column('owner', Integer, ForeignKey('people.id'))
)

meta.create_all(engine)

conn = engine.connect()

"""insert_statement = insert(people).values(name = "Suhail", age = 39)
#insert_statement = people.insert().values(name = "Attif", age = 30)
result = conn.execute(insert_statement)
conn.commit()"""


"""
select_statement = people.select().where(people.c.age > 30)
result = conn.execute(select_statement)

for row in result.fetchall():
    print(row)"""


"""update_statement = people.update().where(people.c.name == 'Suhail').values(age = 50)
result = conn.execute(update_statement)
conn.commit()"""


"""delete_statement = people.delete().where(people.c.name == "Suhail")
result = conn.execute(delete_statement)
conn.commit()"""
