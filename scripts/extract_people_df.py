"""
Usage: python3 extract_people_df.py <matsim_output.xml>

Clément Dauvilliers 13/03/2022 - EPFL TRANSP-OR lab semester project

Extracts all information regarding individuals in the synthetic population in the
MATSIM output for Switzerland.
"""
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import random
import csv
import os
from time import time

# PARAMETERS
# Number of people to process between two saves into the csv file
_SAVE_EVERY_K_PEOPLE_ = 100000
# Save file path
_SAVE_FILE_ = "data/matsim_population_attributes.csv"
# First line in the file to process. Modify if some the file has already
# been parsed in a previous run.
_FIRST_LINE_TO_PROCESS_ = 174427343

_FIELDS_ = ['age', 'bikeAvailability', 'carAvail', 'employed',
            'hasLicense', 'home_x', 'home_y', 'householdIncome',
            'isCarPassenger', 'municipalityType', 'ptHasGA', 'ptHasHalbtax',
            'ptHasStrecke', 'ptHasVerbund', 'sex', 'spRegion']


def refresh_xml_parser(current_parser=None):
    """
    Recreates a new XML parser to free all XML elements stored
    in the current one.
    :param current_parser: the current XML parser being used.
        if None, simply creates a new parser.
    :return: a fresh new XML parser. The new parser has ALREADY
        read the header lines of the document (from the <?xml tag
        to the <population>). It is meant to be directly fed
        <person> elements.
    """
    if current_parser is not None:
        # First: delete the current parser. As it hasn't finished reading
        # the XML file, it is expecting to see a </population> and </xml>
        # tags. Thus the close() method will raise an exception, which we
        # actually don't mind.
        try:
            current_parser.close()
        except ET.ParseError:
            print("Recreating XML parser")

    # Second: create a new parser. Unfortunately, if we directly give the
    # next lines of the XML files (new <person> elements), the parser will raise
    # an exception as it expects a root (like <?xml>).
    # The solution is to feed the new parser with the roots elements (the first few
    # lines of the document) until the <population> tag so that it is ready
    # to receive new <person> elements.
    document_root = '<?xml version="1.0" encoding="utf-8"?>\
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\
<population desc="Switzerland Baseline">'
    new_parser = ET.XMLPullParser(events=('end',))
    new_parser.feed(document_root)
    return new_parser


def main():
    # Counts the number of individuals that have been processed
    processed_people = 0

    # Creates the save file if it doesnt exist yet
    if not os.path.exists(_SAVE_FILE_):
        with open(_SAVE_FILE_, "w", newline='') as savefile:
            writer = csv.DictWriter(savefile, fieldnames=_FIELDS_)
            # Writes the header line of the CSV
            writer.writeheader()

    soc_eco_attrib_rows = []
    start_time = time()
    with open(_SAVE_FILE_, 'a', newline='') as save_file:
        writer = csv.DictWriter(save_file, fieldnames=_FIELDS_)
        with open(sys.argv[1], "r") as matsim_file:
            # Builds the XML non-blocking parser, which will be fed
            # with the successive lines from the matsim file
            xml_parser = refresh_xml_parser()
            for line_number, line in enumerate(matsim_file):
                if line_number < _FIRST_LINE_TO_PROCESS_:
                    continue

                xml_parser.feed(line)
                for event, elem in xml_parser.read_events():
                    # We browse the XML until we find the end of a <person> element.
                    # Then, we retrieve the list of all its attributes
                    # and store their names and values
                    if event == "end" and elem.tag == "person":
                        # We do not count people whose ids are 'freight_xxxxx' as we do not
                        # have socio-economic information about them
                        if elem.attrib['id'][0] == 'f':
                            continue
                        processed_people += 1
                        # The children elements of a <person> are <attributes> and <plan>
                        # The <plan> concerns the activities, which we ignore here
                        attributes = list(elem[0])
                        # att.text contains the value of the attribute
                        # such as "45" for age, or "true" for employed.
                        soc_eco_attrib_rows.append({att.attrib['name']: att.text for att in attributes})

                        if processed_people % 10000 == 0:
                            print(f'Processed {processed_people} people in {time() - start_time}s')

                        # After a fix number of individuals processed, saves their attributes to
                        # the results file
                        if processed_people % _SAVE_EVERY_K_PEOPLE_ == 0:
                            print(f'Saving to {_SAVE_FILE_}, last line processed={line_number}')
                            writer.writerows(soc_eco_attrib_rows)
                            del soc_eco_attrib_rows
                            soc_eco_attrib_rows = []
                            xml_parser = refresh_xml_parser(xml_parser)
    return 0


if __name__ == "__main__":
    main()
