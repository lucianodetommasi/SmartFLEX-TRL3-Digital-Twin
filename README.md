SmartFLEX TRL3 Digital Twin
A Python proof-of-concept of the SmartFLEX user–technology flexibility Digital Twin.

What the code does
The code models flexibility from a flexible energy resource, such as a heat pump, while accounting for user behaviour and uncertainty.

The workflow is:

Technical resource
        ↓
Technical flexibility
        ↓
User constraints
        ↓
User-acceptable flexibility
        ↓
Delivery probability
        ↓
Monte-Carlo simulation
        ↓
Probabilistic flexibility envelope

The model produces:

technical flexibility;

expected flexibility;

P10, P50 and P90 flexibility estimates;

probability of successful delivery;

day-ahead flexibility predictions;

basic prediction-validation metrics.

Main components
FlexibleResource
Defines the technical characteristics of a flexible device:

FlexibleResource(
    name="Heat Pump",
    rated_power_kw=8.0,
    max_flex_fraction=0.75,
    max_duration_hours=2.0,
    response_time_minutes=5,
    recovery_time_hours=1.0
)

UserFlexibilityProfile
Defines behavioural characteristics such as willingness, acceptance, comfort sensitivity and probability of overriding automated control.

SmartFLEXDigitalTwin
Combines the technical resource and user profile.

The main methods are:

technical_flexibility()
user_acceptable_flexibility()
delivery_probability()
monte_carlo_flexibility()
predict_day_ahead()

Monte-Carlo simulation
The Digital Twin runs thousands of possible user/resource responses to represent uncertainty. The resulting simulations are used to calculate the P10, P50 and P90 flexibility envelope.

Validation
The validate_prediction() function compares predicted and observed flexibility using:

MAE;

RMSE;

Normalised MAE.

Running the code
Install dependencies:

pip install numpy pandas matplotlib

Run:

python smartflex_digital_twin.py

The code generates a synthetic 24-hour heat-pump load profile, runs the Digital Twin, displays the flexibility envelope and saves:

smartflex_trl3_flexibility_forecast.csv

The current implementation uses synthetic user and load data. These can later be replaced with real questionnaire, equipment and pilot-site data.
