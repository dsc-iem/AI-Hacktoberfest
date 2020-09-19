# Forest Fire Detection


```python
Hello World ! 

from AI import ForestFirePredictions
```

Welcome to the Hacktoberfest Challenge for Artificial Intelligence | Machine Learning !

Today we will be assessing your skills to Predict Forest Fire Areas given its various parameters!

## Features Provided
For this dataset you are provided with the following features/attributes of a park:

X - x-axis spatial coordinate within the park
Y - y-axis spatial coordinate within the park
month - month of the year: "jan" to "dec"
day - day of the week: "mon" to "sun"

FFMC - FFMC index from the FWI system: 18.7 to 96.20 : The Fine Fuel Moisture Code (FFMC) represents fuel moisture of forest litter fuels under the shade of a forest canopy. It is intended to represent moisture conditions for shaded litter fuels, the equivalent of 16-hour timelag. It ranges from 0-101. Subtracting the FFMC value from 100 can provide an estimate for the equivalent (approximately 10h) fuel moisture content, most accurate when FFMC values are roughly above 80.

DMC - DMC index from the FWI system: 1.1 to 291.3 : The Duff Moisture Code (DMC) represents fuel moisture of decomposed organic material underneath the litter. System designers suggest that it is represents moisture conditions for the equivalent of 15-day (or 360 hr) timelag fuels. It is unitless and open ended. It may provide insight to live fuel moisture stress.


DC - DC index from the FWI system: 7.9 to 860.6 : The Drought Code (DC), much like the Keetch-Byrum Drought Index, represents drying deep into the soil. It approximates moisture conditions for the equivalent of 53-day (1272 hour) timelag fuels. It is unitless, with a maximum value of 1000. Extreme drought conditions have produced DC values near 800.

ISI - ISI index from the FWI system: 0.0 to 56.10 : The Initial Spread Index (ISI) is a numeric rating of the expected rate of fire spread. It is based on wind speed and FFMC. Like the rest of the FWI system components, ISI does not take fuel type into account. Actual spread rates vary between fuel types at the same ISI.

temp - temperature in Celsius degrees: 2.2 to 33.30
RH - relative humidity in %: 15.0 to 100
wind - wind speed in km/h: 0.40 to 9.40
rain - outside rain in mm/m2 : 0.0 to 6.4
area - the burned area of the forest (in hectares): 0.00 to 1090.84
(this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform).

More Technical Data about the features could be found out at [Fire Weather Index System](https://www.nwcg.gov/publications/pms437/cffdrs/fire-weather-index-system) & [FWI Canada](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi).

## Submission Criteria
anybody part of the DCS-Iem Group is eligible to contribute to this Open Source Project.

## Submission Procedure


## Credits
The credits for this repository is hidden and will be provided post event completion/evaluation.
