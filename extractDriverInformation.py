import csv

def getDriverName(state, license_plate):
    """
    Get the driver's name associated with the
    given license plate
    """
    # organize all driver information
    driver_info = {}
    with open('driverInfo.csv', newline='') as csv_file:
        for row in csv.reader(csv_file, delimiter=','):
            driver_info[(row[0],row[1])] = row[2]
    
    # return driver's first and last name
    return driver_info[(state, license_plate)]
