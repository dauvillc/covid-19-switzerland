"""
Usage: python3 extract_people_df.py <matsim_output.xml>

Clément Dauvilliers 13/03/2022 - EPFL TRANSP-OR lab semester project

Extracts all information regarding individuals in the synthetic population in the
MATSIM output for Switzerland.
"""
import pandas as pd
import xml.etree.ElementTree as ET
import sys


def main():
    count = 0
    NUM_PERSONS_TO_ADD_TO_DB = 10  # This will just export the N first persons in the xml to the databas

    soc_eco_attrib_rows = []
    for event, elem in ET.iterparse(sys.argv[1], events=("start", "end")):
        # Early stopping condition
        if count == NUM_PERSONS_TO_ADD_TO_DB:
            break
        # We browse the XML until we find the end of a <person> element.
        # Then, we retrieve the list of all its attributes
        # and store their names and values
        if event == "end" and elem.tag == "person":
            count += 1
            # The children elements of a <person> are <attributes> and <plan>
            # The <plan> concerns the activities, which we ignore here
            attributes = list(elem[0])
            # att.text contains the value of the attribute
            # such as "45" for age, or "true" for employed.
            soc_eco_attrib_rows.append([att.text for att in attributes])

    features = ['age', 'bikeAvailability', 'carAvailability', 'employed', 'hasLicense',
                'home_x', 'home_y', 'householdIncome', 'isCarPassenger', 'municipalityType',
                'ptHasGA', 'ptHasHalbtax', 'ptHasStrecke', 'ptHasVerbund', 'sex', 'spRegion']
    population_df = pd.DataFrame(soc_eco_attrib_rows, columns=features)
    print(population_df)


if __name__ == "__main__":
    main()
