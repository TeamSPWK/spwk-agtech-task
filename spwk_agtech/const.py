OBSERVATIONS = {
    'DVS': {
    # 'crop DeVelopment Stage' 0 at emergence 1 at Anthesis (flowering) and 2 at maturity / 0, 1, 2
        'min': 0,
        'max': 2,
        'unit':'stage',
        'mean':'Development stage'},
    'LAI': {
    # 'Leaf Area Index' including stem and pod area / 
        'min': 0,
        'max': 10,
        'unit':'ha/ha',
        'mean':'LAI'},
    'TAGP': {
    # 'Total Above Ground Production' / kg ha-1
        'min': 105,
        'max': 30000,
        'unit':'kg/ha',
        'mean':'Biomass'},
    'TWSO': {
    # 'Total dry Weight of Storage Organs' / kg ha-1
        'min': 0,
        'max': 11000,
        'unit':'kg/ha',
        'mean':'Yield'},
    'TWLV': {
    # 'Total dry Weight of LeaVes' / kg ha-1
        'min': 68.25,
        'max': 7500,
        'unit':'kg/ha',
        'mean':'Leaves DW'},
    'TWST': {
    # 'Total dry Weight of STems' / kg ha-1
        'min': 36.75,
        'max': 12500,
        'unit':'kg/ha',
        'mean':'Stems DW'},
    'TWRT': {
    # 'Total dry Weight of RooTs' / kg ha-1
        'min': 105,
        'max': 4500,
        'unit':'kg/ha',
        'mean':'Roots DW'},
    'TRA': {
    # 'crop TRAnspiration RAte' / cm day-1
        'min': 0,
        'max': 2,
        'unit':'cm/day',
        'mean':'Transpiration rate'},
    'RD': {
    # 'Rooting Depth' / cm
        'min': 10,
        'max': 120,
        'unit':'cm',
        'mean':'Root depth'},
    'SM': {
    # 'Soil Moisture' root-zone, Volumetric moisture content in root zone /
        'min': 0.3,
        'max': 0.57,
        'unit':'cm3/cm3',
        'mean':'Soil moisture'},
    'WWLOW': {
    # 'WWLOW = WLOW + W' Total amount of water in the soil profile / cm
        'min': 54.177,
        'max': 68.5,
        'unit':'cm',
        'mean':'Water in soil'}
}

ACTIONS = {
    'IRRAD': {
    # Incoming global radiaiton
        'min': 0,
        'max': 4e7,
        'unit':'J/m2/day'},
    'TMIN': {
    # Daily minimum temperature 
        'min': -50,
        'max': 60,
        'unit':'Celsius'},
    'TMAX': {
    # Daily maximum temperature
        'min': -50,
        'max': 60,
        'unit':'Celsius'},
    'VAP': {
    # Daily mean vapour pressure
        'min': 0.06 + 1e-5,
        'max': 199.3 - 1e-4,
        'unit': 'hPa'},
    'RAIN': {
    # Daily total rainfall
        'min': 0,
        'max': 25,
        'unit': 'cm/day'},
    'E0': {
    # Penman potential evaporation from a free water surface
        'min': 0,
        'max': 2.5,
        'unit':'cm/day'},
    'ES0': {
    # Penman potential evaporation from a moist bare soil surfac
        'min': 0,
        'max': 2.5,
        'unit':'cm/day'},
    'ET0': {
     # Penman or Penman-Monteith potential evaporation for a reference crop canopy
        'min': 0,
        'max': 2.5,
        'unit':'cm/day'},
    'WIND': {
    # Daily mean wind speed at 2m height
        'min': 0,
        'max': 100,
        'unit':'m/sec'},
    'IRRIGATE': {
    #  Amount of irrigation in cm water applied on this day.
        'min': 0,
        'max': 50,
        'unit':'cm'},
    'N': {
    # Amount of N fertilizer in kg/ha applied on this day.
        'min': 0,
        'max': 100,
        'unit':'kg/ha'},
    'P': {
    # Amount of P fertilizer in kg/ha applied on this day.
        'min': 0,
        'max': 100,
        'unit':'kg/ha'},
    'K': {
    # Amount of K fertilizer in kg/ha applied on this day.
        'min': 0,
        'max': 100,
        'unit':'kg/ha'}    
}