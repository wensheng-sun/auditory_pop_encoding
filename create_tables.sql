USE auditorycortex;
CREATE TABLE unitinfo
(
    unitid int unsigned NOT NULL auto_increment, # unique identifier of a unit(meaning neuron)
    unitnumber int unsigned NOT NULL, # the unit identifer within all neurons from an animal
    monkey char(4) NOT NULL, # the id of the monkey from which the unit is recorded
    file char(4) NOT NULL, # the file number that stored the data of the unit
    chan tinyint NOT NULL, # channel number of the unit
    PRIMARY KEY (unitid)
);

CREATE TABLE stimulusinfo
(
    stimulusid int unsigned NOT NULL PRIMARY KEY, # unique identifier of the stimulus
    frequency decimal NOT NULL, # frequency in (Hz) of the stimulus
    attenuation decimal NOT NULL # sound attenuation in dB for the stimulus
);

CREATE TABLE actionpotential
(
    unitid int unsigned NOT NULL, # unit that fires the action potential (AP)
    stimulusid int unsigned NOT NULL, # the sound stimulus presented during the recording
    repetition int unsigned NOT NULL, # repetition number of the recording
    timeofevents text NOT NULL, # time of events, a string of comma separated numbers  
    recommended bit NOT NULL, # whether the data should be used based on quality 1=Y,0=N
    
    PRIMARY KEY (unitid, stimulusid, repetition), # unitid and trialid together uniquely defines a record
    
    FOREIGN KEY (unitid) REFERENCES unitinfo (unitid) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (stimulusid) REFERENCES stimulusinfo (stimulusid) ON UPDATE CASCADE ON DELETE RESTRICT
);

