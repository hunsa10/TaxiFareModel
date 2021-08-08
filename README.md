# TaxiFareModel
TaxiFareModel predicts the price of a taxi trip from a given pickup location to
a specified dropoff location, takes into account the date, time and number of
passengers on the trip. The model is trained and tested on data from the
New York City Taxi Fare Prediction dataset (https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data).

The model calculates the haversine distance between pickup and dropoff location
and uses a linear regression to predict a final taxi fare based on distance
between pickup and dropoff location, date, time and number of passengers.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for TaxiFareModel in gitlab.com/hunsa10.
If your project is not set please add it:

- Create a new project on `gitlab.com/hunsa10/TaxiFareModel`
- Then populate it:

```bash
git remote add origin git@github.com:hunsa10/TaxiFareModel.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```

# Install

Go to `https://github.com/hunsa10/TaxiFareModel` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:hunsa10/TaxiFareModel.git
cd TaxiFareModel
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
TaxiFareModel-run
```
