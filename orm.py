from sqlalchemy import create_engine, Integer, String, Float, Column,ForeignKey

# declarative_base is an alternative to metadata
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

engine = create_engine("mysql+pymysql://root:Code%407338677189@localhost:3306/demo", echo = True)

Base = declarative_base()

class Person(Base):
    __tablename__ = 'person'
    id = Column(Integer, primary_key = True)
    name = Column(String(30), nullable=False)
    age = Column(Integer)

    things = relationship('Thing', back_populates='person')

class Thing(Base):
    __tablename__ = 'things'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    description = Column(String(50), nullable=False)
    value = Column(Float)
    owner = Column(Integer, ForeignKey('person.id'))

    person = relationship('Person', back_populates='things')

Base.metadata.create_all(engine)

Session = sessionmaker(bind = engine)
session = Session()

"""new_person = Person(name = 'Charlie', age = 70)
session.add(new_person)
session.flush() # add the data to the database but is not permanent untill commit can be rollbacked


new_thing = Thing(description = 'Camera', value = 400, owner = new_person.id)
session.add(new_thing)

session.commit()"""

#result = session.query(Person.name, Person.age).all() # alternative to fetchall()
"""result = session.query(Person).filter(Person.age > 40).all()
print([p.name for p in result])"""

result = session.query(Thing).filter(Thing.value > 300).delete()
session.commit()

result = session.query(Thing).filter(Thing.value < 50).all()
print([t.description for t in result])

# same for update
# for grouping also import func

#result = session.query(Thing.owner, func.sum(Thing.value)).group_by(Thing.owner).having(func.sum(Thing.value)>50).all(
#print(result)

session.close()