"""
Usage: python3 extract_people_df.py <matsim_output.xml>

Clément Dauvilliers 13/03/2022 - EPFL TRANSP-OR lab semester project

Extracts all information regarding individuals in the synthetic population in the
MATSIM output for Switzerland.
"""
import xml.etree.ElementTree as ET
import sys
import csv
import os
from time import time

# PARAMETERS
# Whether to process the attributes of people
_PROCESS_PERSON_ATTRIBUTES_ = False
# Number of people to process between two saves into the csv file
# A lower number implies saving more often and thus a slight loss
# of time (the saving time is still small compared to the processing
# time) but requires less RAM.
# For example, 100,000 usually demands about 6GB of memory
_SAVE_EVERY_K_PEOPLE_ = 100000
# Save file paths
_SAVE_FILE_ATTRIBUTES_ = "data/matsim_population_attributes.csv"
_SAVE_FILE_ACTIVITIES_ = "data/matsim_population_activities.csv"
# First line in the file to process. Modify if some the file has already
# been parsed in a previous run.
_FIRST_LINE_TO_PROCESS_ = 0

_FIELDS_ATTRIBUTES_ = ['id', 'age', 'bikeAvailability', 'carAvail', 'employed',
                       'hasLicense', 'home_x', 'home_y', 'householdIncome',
                       'isCarPassenger', 'municipalityType', 'ptHasGA', 'ptHasHalbtax',
                       'ptHasStrecke', 'ptHasVerbund', 'sex', 'spRegion']

_FIELDS_ACTIVITIES_ = ['id', 'type', 'facility', 'link', 'x', 'y', 'start_time', 'end_time']


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
    # Create a new parser.
    new_parser = ET.XMLPullParser(events=('end',))
    if current_parser is not None:
        # Delete the current parser. As it hasn't finished reading
        # the XML file, it is expecting to see a </population> and </xml>
        # tags. Thus the close() method will raise an exception, which we
        # actually don't mind.
        try:
            current_parser.close()
        except ET.ParseError:
            print("Recreating XML parser")

        # Unfortunately, if we directly give the
        # next lines of the XML files (new <person> elements), the parser will raise
        # an exception as it expects a root (like <?xml>).
        # The solution is to feed the new parser with the roots elements (the first few
        # lines of the document) until the <population> tag so that it is ready
        # to receive new <person> elements.
        document_root = '<?xml version="1.0" encoding="utf-8"?>\
        <!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\
        <population desc="Switzerland Baseline">'
        new_parser.feed(document_root)
    return new_parser


def create_csv_file(path, fieldnames):
    """
    Creates a CSV file at the given path and writes
    its header. If the file already exists, does
    nothing.
    :param path: where to create the csv;
    :param fieldnames: list of names.
    :return: (file, writer) where:
        - file is the python File object used to write into the file.
        - writer is the csv.DictWriter object used to write the CSV.
    """
    if os.path.exists(path):
        file = open(path, "a", newline='')
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
    else:
        file = open(path, "w", newline='')
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
    return file, writer


def main():
    # Counts the number of individuals that have been processed
    processed_people = 0

    # Creates the save files if they dont exist yet
    attributes_csv, attributes_writer = create_csv_file(_SAVE_FILE_ATTRIBUTES_, _FIELDS_ATTRIBUTES_)
    activities_csv, activities_writer = create_csv_file(_SAVE_FILE_ACTIVITIES_, _FIELDS_ACTIVITIES_)

    # Lists containing the last rows that have been extracted from the XML
    # but have yet to be written into the CSV file
    soc_eco_attrib_rows = []
    last_activities = []

    start_time = time()
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
                    person_id = elem.attrib['id']
                    if person_id[0] == 'f':
                        continue
                    processed_people += 1

                    # ==== PROCESSING ATTRIBUTES ========== #
                    if _PROCESS_PERSON_ATTRIBUTES_:
                        # The children elements of a <person> are <attributes> and <plan>
                        # The <plan> concerns the activities, which we ignore here
                        attributes = list(elem[0])
                        # att.text contains the value of the attribute
                        # such as "45" for age, or "true" for employed.
                        attributes = {att.attrib['name']: att.text for att in attributes}
                        attributes['id'] = person_id
                        soc_eco_attrib_rows.append(attributes)

                    # ==== PROCESSING ACTIVITIES =========== #
                    # elem[1] is the <plan> element. Its children can
                    # be <leg> or <activity>
                    activities = [xmlelem for xmlelem in elem[1] if xmlelem.tag[0] == 'a']
                    # Compiles the attributes (type, facility, x, y, ...) of all activities
                    # of the current person.
                    person_activities = [
                        {name: value for name, value in activity.attrib.items()}
                        for activity in activities
                    ]
                    # Adds the ID of the current person to that information, then
                    # add all of it to last_activities so that they'll be written into the
                    # results file
                    for activity_data in person_activities:
                        activity_data['id'] = person_id
                    last_activities += person_activities

                    if processed_people % 10000 == 0:
                        print(f'Processed {processed_people} people in {time() - start_time}s')

                    # After a fix number of individuals processed, saves their attributes
                    # and activities
                    if processed_people % _SAVE_EVERY_K_PEOPLE_ == 0:
                        # Write the attributes
                        if _PROCESS_PERSON_ATTRIBUTES_:
                            print(f'Saving attributes to {_SAVE_FILE_ATTRIBUTES_}, last line processed={line_number}')
                            attributes_writer.writerows(soc_eco_attrib_rows)
                        # Write the activities
                        activities_writer.writerows(last_activities)
                        print(f'Saving activities to {_SAVE_FILE_ACTIVITIES_}, last line processed={line_number}')

                        # Frees all memory used so far to store attributes and activities
                        del soc_eco_attrib_rows
                        del last_activities
                        soc_eco_attrib_rows = []
                        last_activities = []
                        # Recreates the XML parser to free the memory, otherwise
                        # all XML seen so far is still remembered and kept in mem
                        xml_parser = refresh_xml_parser(xml_parser)

    # Close the CSV files before exitting
    activities_csv.close()
    attributes_csv.close()
    return 0


if __name__ == "__main__":
    main()
