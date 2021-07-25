# Python 3.8 Version of pile_monitor application

import os
from pathlib import Path
import fnmatch
import configparser
import time
from collections import namedtuple
import datetime as dt
import pandas as pd
from pandas import ExcelWriter
import numpy as np
from scipy.io import wavfile
import scipy.io

# Use Peak namedtuple to capture info about either a peak in an impulse strike
# imp file or a one second timespan in vibratory vib file """
Peak = namedtuple(
    "Peak", ["filename", "ix", "ts", "time", "peakdb", "sel90db", "rms90db", "datetime"]
)

# Utility functions for unit conversion ----
def db2pa(db):
    return 1e-6 * 10 ** (s / 20)


def pa2db(pa):
    if pa > 0:
        return 20 * np.log10(pa / 1e-6)
    else:
        return 0


def sumdb(s):
    """ In: a pandas series of db values;
    converts to pa, sum all
    Out: reconvert to db"""
    spa = 1e-6 * 10 ** (s / 20)
    sumpa = spa.sum()
    return pa2db(sumpa)


def csel(s):
    """ same as sumdb; used as agg in pivot table"""
    spa = 1e-6 * (10 ** (s / 10))
    sumpa = spa.sum()
    sum_db = 10 * np.log10(sumpa / 1e-6)
    # print(sum_db)
    return sum_db


def meandb(s):
    """" In: a pandas series of db values;
    converts to pa, mean all 
    Out: reconvert to db"""
    spa = 1e-6 * 10 ** (s / 20)
    meanpa = spa.mean()
    return pa2db(meanpa)


def nrStrikes(s):
    """" In: a pandas series of db values
    Out: nr of elements; only consider peaks with ix > 0"""
    s_gt_0 = s[s > 0.0]
    return s_gt_0.size


def ms2hms(millis):
    seconds = (millis / 1000) % 60
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    return "%d:%d:%5.3f" % (hours, minutes, seconds)


def s2hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%5.3f" % (h, m, s)


def ix2hms(ix, samplerate):
    secs = ix / samplerate
    return s2hms(secs)


# Main functions ----
def peak_search(splpa, minpeakdb, lookaheadrows, samplerate, interval_between_strikes):
    minpeakpa = 1e-6 * 10 ** (minpeakdb / 20)

    lookaheadtime = lookaheadrows / samplerate
    lookaheadtimedb = 10 * np.log10(lookaheadtime)

    sliderows = int(interval_between_strikes * samplerate / 2)
    # speedup: search in mask for True for each ix with splpa[ix] >= minpeakpa
    mask_minpeakpa = splpa >= minpeakpa
    peakixlst = []
    nextabsoluteix = 0
    len_spla = len(splpa)
    while nextabsoluteix < len_spla:
        peakrelativeix = np.argmax(mask_minpeakpa[nextabsoluteix:])

        if peakrelativeix == 0:  # argmax can't find value >= minpeak
            if nextabsoluteix == 0 and splpa[0] < minpeakpa:
                break  # first value found too soft
            if nextabsoluteix > 0:
                break  # no further values > min found
        peakixlst.append(nextabsoluteix + peakrelativeix)
        nextabsoluteix = nextabsoluteix + peakrelativeix + sliderows

    return peakixlst


def list_peaks(peakixlst, splpa, samplerate, filename, datetime):
    peaklist = []
    p10_lst = np.zeros(len(peakixlst))
    p10_i = 0
    for peakix, start in enumerate(peakixlst):
        if start >= 0:
            # peakixlst[nextpeakix] is the wav start of the next peak if any
            nextpeakix = peakix + 1

            if nextpeakix < len(peakixlst):
                lenpeak = peakixlst[nextpeakix] - start
            else:
                lenpeak = samplerate  # approximate value ~ 1 second

            slice100pa = splpa[start : start + lenpeak - 1]
            p10 = 0.1 * max(slice100pa)
            # print(f'10% of peak dB: {pa2db(p10)}')
            p10_loc = np.where(slice100pa >= p10)
            t1 = start + p10_loc[0][0]
            t2 = start + p10_loc[0][-1]
            slice90pa = splpa[t1:t2]
            rms90pa = np.sqrt(np.mean(slice90pa ** 2))
            rms90db = pa2db(rms90pa)
            sel90db = rms90db + 10 * np.log10(len(slice90pa) / samplerate)
            peak90ix = np.argmax(slice90pa)
            peak_splpa = np.max(slice90pa)
            peak_spldb = pa2db(peak_splpa)
            # print(f"Peak dB: {peak_spldb:.2f}")
            ix_from_start = start + peak90ix
            time_seconds_from_start = ix_from_start / samplerate
            time1 = s2hms(time_seconds_from_start)

            # print(f"P90 length: {len(slice90pa) / samplerate * 1000:.3f} ms")

            peak = Peak(
                filename=filename,
                ix=start,
                ts=time_seconds_from_start,
                time=time1,
                peakdb=peak_spldb,
                sel90db=sel90db,
                rms90db=rms90db,
                datetime=datetime,
            )

            p10_lst[p10_i] = p10
            p10_i += 1
        else:
            peak = Peak(
                filename=filename,
                ix=0,
                ts=0,
                time=0,
                peakdb=0,
                sel90db=0,
                rms90db=0,
                datetime=datetime,
            )

        peaklist.append(peak)

    min_db = pa2db(np.min(p10_lst))

    if min_db <= 150:
        min_db = 150

    return min_db, peaklist


def pile_peak_lst(
    filename,
    splpa,
    spldb,
    minpeakdb,
    pct_maxdb,
    lookaheadrows,
    interval_between_strikes,
    samplerate,
    datetime,
):
    """returns [[startix,stopix],..] of peak segments in the pile strike time
    series. Algorithm: search for padb > minpeakdb (150db);
    then search for max padb value in the next 50ms; this is the peak; store
    peak in peakixlst;  now jump forward 1 second; repeat peak search.
    When done create start-stop list; process this list and compute peak, rms,
    sel for each peak and append to peaklist, a namedtuple list of Peaks.
    typical values: lookaheadtime = 0.050 seconds, interval_between_strikes = 1.3 seconds.
    lookaheadrows = int(lookaheadtime * samplerate)
    Return this peaklist.
    """
    start_time = time.process_time()

    lookaheadtime = lookaheadrows / samplerate

    # First iteration with cfg minpeakdb
    peakixlst = peak_search(
        splpa, minpeakdb, lookaheadrows, samplerate, interval_between_strikes
    )

    # peaklist contains Peak tuples; peakixlst contains ticks of peak starts

    p10_db, peaklist = list_peaks(peakixlst, splpa, samplerate, filename, datetime)

    if p10_db > 150:
        # Second iteration with smallest 10% max peak
        print(f"Second iteration with {p10_db:.3f} dB minimum peak.")
        peakixlst = peak_search(
            splpa, p10_db, lookaheadrows, samplerate, interval_between_strikes
        )

    peaklist = list_peaks(peakixlst, splpa, samplerate, filename, datetime)[1]

    return p10_db, peaklist


def compute_imp_wav_file_stats(
    filename,
    hydr_sens,
    minpeakdb,
    pct_maxdb,
    filename_minpeakdb_dict,
    lookaheadtime,
    interval_between_strikes,
    datetime,
):
    """read wav 24bit pcm file that contains integer Pa values of impulse pile driving sound levels.
    Convert Pa (Sound Level) to muPa^2 db (SPL dB re:muPa) values for further processing.Compute stats for each strike.
    New design: Add a column ts rounded to whole seconds. Use pivot on this column to compute stats"""

    filename_short = os.path.basename(filename)

    samplerate, splpa = wavfile.read(filename)
    splpa = splpa // 255

    todo = []

    calibration_factor = hydrophone_calib(hydr_sens, samplerate)

    peaklist_imp = []
    if splpa.ndim == 2:  # two sound channels; ndarray has two columns;
        if np.abs(splpa[:, 0:1]).sum() >= np.abs(splpa[:, 1:2]).sum():
            todo = [
                [filename_short + ".1.loud", splpa[:, 0:1]],
                [filename_short + ".2.soft", splpa[:, 1:2]],
            ]
        else:
            todo = [
                [filename_short + ".2.loud", splpa[:, 1:2]],
                [filename_short + ".1.soft", splpa[:, 0:1]],
            ]
    else:
        todo = [[filename_short, splpa]]

    for filename_short, splpa1 in todo:
        lookaheadrows = int(lookaheadtime * samplerate)
        # convert raw wav values to muPa^2 units
        # calibration_factor = 1.0  # find calibration by running with 1.0 and comparing peak results with S+ peaks
        splpa1 = np.abs(splpa1 / calibration_factor)
        # (splpa1 / calibration_factor * 10 ** 6) ** 2
        splpa1[splpa1 <= 0] = 1e-6  # to accomodate log10
        spldb = 20 * np.log10(splpa1 / 1e-6)
        if pct_maxdb:  # if 0 then use given minpeakdb
            if pct_maxdb * spldb.max() <= 150:
                minpeakdb = 150
            else:
                minpeakdb = pct_maxdb * spldb.max()
            # minpeakdb = pct_maxdb * spldb.max()  # 0.9
            filename_minpeakdb_dict[filename_short] = minpeakdb

        print(f"Initial minimum peak: {minpeakdb:.3f} dB")
        min_db, peaklist_sub = pile_peak_lst(
            filename_short,
            splpa1,
            spldb,
            minpeakdb,
            pct_maxdb,
            lookaheadrows,
            interval_between_strikes,
            samplerate,
            datetime,
        )
        filename_minpeakdb_dict[filename_short] = round(min_db, 3)
        peaklist_imp.extend(peaklist_sub)

    return peaklist_imp


def compute_vib_wav_file_stats(filename, hydr_sens, datetime):
    """read wav 24bit pcm file that contains raw, uncalibrated integer Pa values
    of vibratory pile driving sound levels. Convert raw to calibrated Pa,
    then to muPa^2 db (SPL dB re: muPa) values for further processing.
    Compute stats for 1 second segments.
    """
    filename_short = os.path.basename(filename)
    samplerate, data = wavfile.read(filename)
    nrows = nrseconds = data.size // samplerate
    todo = []

    calibration_factor = hydrophone_calib(hydr_sens, samplerate)

    peaklist_vib = []
    if data.ndim == 2:  # two sound channels; ndarray has two columns;
        if np.abs(data[:, 0:1]).sum() >= np.abs(data[:, 1:2]).sum():
            todo = [
                [filename_short + ".1.loud", data[:, 0:1]],
                [filename_short + ".2.soft", data[:, 1:2]],
            ]
        else:
            todo = [
                [filename_short + ".2.loud", data[:, 1:2]],
                [filename_short + ".1.soft", data[:, 0:1]],
            ]
    else:
        todo = [[filename_short, data]]

    # TODO: replace for loop with pivot on new secs col derived from int(ix/samplerate)
    for filename_short, data1 in todo:
        for nrow in range(nrows):
            startix = nrow * samplerate
            nextix = startix + samplerate
            splpa = data1[startix:nextix]
            splpa = np.abs(splpa / calibration_factor)
            # convert raw SLPa to muPa^2
            splpa[splpa == 0] = 1e-6
            spldb = 20 * np.log10(splpa / 1e-6)
            rmspa = np.sqrt(np.mean(splpa ** 2))
            # rmspa2= np.average(splpa)  # average of each row, which is a 1-second timeseries
            rmsdb = 20 * np.log10(rmspa / 1e-6)
            # rmsdb2 = 20 * np.log10(rmspa2/1e-6)
            peakdb = np.max(spldb)
            time1 = s2hms(nrow)
            peak = Peak(
                filename=filename_short,
                ix=startix,
                ts=nrow,
                time=time1,
                peakdb=peakdb,
                sel90db=rmsdb,
                rms90db=rmsdb,
                datetime=datetime,
            )
            peaklist_vib.append(peak)

    return peaklist_vib


def hydrophone_calib(sens, rate=48000):
    """microphone calib is -198.750. I computed calibration factors by
    comparing raw wav file pa values with pa values shown by Spectrum+
    or look at the file in the S+/cal idrectory:
    hydrophone-198_750.cal
    TransducerSensitivityVolts=-198.750000 (dB re V/uPa)
    DAQVoltageRange=10.000000
    CalRatioLeft=86596.429688		daq v range * V/Pa computed from the sensitivity (10^(sensitivity/20)) * 10^6
    calibration factor=2^23/CalRatioLeft
    2^23 is the max raw value for 24 bits wav data (1 bit for sign, 23 bits for integer values)
    """
    bitmax = 2 ** 23

    if rate == 48000:
        daq_voltage = 10.000
    elif rate == 96000:
        daq_voltage = 2.500
    else:
        rate = 48000
        daq_voltage = 10.000

    cal_ratio = daq_voltage * (10 ** (sens / 20)) * 1e-6
    cal_factor = bitmax / cal_ratio

    return cal_factor


def report_error(error_msg):
    return 0


def output_excel(strikes_df, summary_df, settings_df, report_fn):
    writer = ExcelWriter(report_fn, engine="xlsxwriter")
    workbook = writer.book

    strikes_df.to_excel(writer, "Strikes")
    worksheet_strikes = writer.sheets["Strikes"]
    format1 = workbook.add_format({"num_format": "0.00"})
    worksheet_strikes.set_column("B:Z", 12, format1)
    worksheet_strikes.set_column("A:A", 45, None)

    summary_df.to_excel(writer, "Summary", index=False)
    worksheet_summary = writer.sheets["Summary"]
    worksheet_summary.set_column("B:Z", 10, format1)
    worksheet_summary.set_column("A:A", 45, None)

    settings_df.to_excel(writer, "Settings", index=True)
    worksheet_settings = writer.sheets["Settings"]
    worksheet_settings.set_column("A:A", 45, None)

    try:
        writer.save()
    except IOError:
        error_msg = f"WARNING: Please close {dirnameshort}.xlsx!"
        print(error_msg)
        report_error(error_msg)

    return 0


def process_files(
    filenamelst,
    minpeakdb,
    pct_maxdb,
    interval_between_strikes,
    hydr_sens,
    run_settings,
    filename_minpeakdb_dict,
    str_signal=None,
    progress_signal=None,
):

    lookaheadtime = run_settings["lookaheadtime"]  # 0.050

    filename = filenamelst[0]
    filename_short = os.path.basename(filename)
    filename_path = os.path.split(filename)[0]  # "c:/a/b"
    dirnameshort = os.path.basename(os.path.dirname(filename))  # "080217"
    datetime = time.strftime("%Y-%m-%d_%H%M")

    peaklist = []
    print(
        f"Processing {len(filenamelst):d} wav files in dir: {dirnameshort}\n"
    )  # onscreen progress
    filetotal = len(filenamelst)
    for filenr, filename in enumerate(filenamelst):
        datetime1 = time.strftime("%Y-%m-%d_%H%M")
        if "vib" in filename:
            # Note: Lowercase vib only!
            peaklist.extend(compute_vib_wav_file_stats(filename, hydr_sens, datetime))
        else:
            peaklist_elm = compute_imp_wav_file_stats(
                filename,
                hydr_sens,
                minpeakdb,
                pct_maxdb,
                filename_minpeakdb_dict,
                lookaheadtime,
                interval_between_strikes,
                datetime,
            )
            # pprint.pprint(peaklist_elm)
            if not peaklist_elm:
                print(f"{os.path.basename(filename)}: No Peaks detected.\n")
                continue
            peaklist.extend(peaklist_elm)

        print(f"{filenr + 1}\\{filetotal}, { os.path.basename(filename)}\n")
        if str_signal:
            str_signal.emit(f"{filenr + 1}\\{filetotal}, { os.path.basename(filename)}")
        if progress_signal:
            progress_signal.emit(
                round(100 * (filenr + 1.0) / filetotal, 0)
            )  # progressbar expects 1..100

    # print("peaklist", peaklist)
    if not peaklist:
        print("\nNo files with peaks detected.\n")
        return

    dfdb = pd.DataFrame(peaklist, columns=peaklist[0]._fields)
    dfdb.set_index(["filename"], inplace=True)

    report_xlsx_filename = f"{filename_path}/report_{dirnameshort}.xlsx"
    datetime = time.strftime("%Y-%m-%d_%H%M")
    alt_report_xlsx_filename = f"{filename_path}/report_{dirnameshort}_{datetime}.xlsx"

    dfdb_round = dfdb.round(decimals=2)

    # refer to jupyter notebook pile_summary_rpts.ipynb
    # dfdb = pd.read_excel('report_pilestrike_analysis.xlsx')

    run_settings.update(filename_minpeakdb_dict)
    # print(f"Run settings:\n{filename_minpeakdb_dict}")

    # output_excel(dfdb_round, pt_with_total, settings_df, report_xlsx_filename)

    # print("Done. Created a report*.xlsx file.")
    # if str_signal:
    #     str_signal.emit("Done. Created a report*.xlsx and a summary.csv file.")
    # if progress_signal:
    #     progress_signal.emit(0)

    # # ptsummary_lst = ptsummary.T.reset_index().values.T.tolist()
    # # https://stackoverflow.com/questions/49176376/pandas-dataframe-to-lists-of-lists-including-headers
    return dfdb, dfdb_round, report_xlsx_filename


def summaries(dfdb, run_settings):
    pivot_values = ["peakdb", "sel90db", "rms90db"]
    aggfunc = {
        "peakdb": [nrStrikes, meandb, max],
        "sel90db": [csel, meandb],
        "rms90db": [meandb, max],
    }
    dfdb.reset_index(inplace=True)

    # print(dfdb.tail(10))

    pt = pd.pivot_table(dfdb, index="filename", values=pivot_values, aggfunc=aggfunc)
    pt.columns = ["_".join(col).strip() for col in pt.columns.values]
    pt.reset_index(inplace=True)  # move index filename to col filename

    dfdb_loud_totals = dfdb[~dfdb["filename"].str.contains("soft")]
    dfdb_loud_totals = dfdb_loud_totals.copy()
    dfdb_loud_totals[
        "filename"
    ] = " Total"  # set to " Total" with space as first char to force correct pandas sort_values
    dfdb_loud_totals.set_index("filename", inplace=True)

    pt_total = pd.pivot_table(
        dfdb_loud_totals, index="filename", values=pivot_values, aggfunc=aggfunc
    )
    pt_total.columns = ["_".join(col).strip() for col in pt_total.columns.values]
    pt_total.reset_index(inplace=True)

    pt_with_total = pd.concat([pt, pt_total], ignore_index=True)
    # filename	peakdb_max	peakdb_meandb	peakdb_nrStrikes	rms90db_max	rms90db_meandb	seldb_csel	seldb_meandb
    pt_with_total = pt_with_total[
        [
            "filename",
            "peakdb_nrStrikes",
            "peakdb_max",
            "peakdb_meandb",
            "sel90db_csel",
            "sel90db_meandb",
            "rms90db_max",
            "rms90db_meandb",
        ]
    ]
    pt_with_total = pt_with_total.round(decimals=2)

    settings_df = pd.DataFrame.from_dict(run_settings, orient="index").rename(
        columns={0: "setting"}
    )

    return pt_with_total, settings_df


def process_dir(
    rootdir, already_processed, all=False, str_signal=None, progress_signal=None
):
    parser = configparser.ConfigParser()
    cfg_file = Path(rootdir, "monitor.cfg")
    if cfg_file.is_file():
        # print("I exist!")
        parser.read(cfg_file.resolve())

        minpeakdb = parser.getfloat("pile", "minpeakdb")
        interval_between_strikes = parser.getfloat("pile", "interval_between_strikes")
        lookaheadtime = parser.getfloat("pile", "lookaheadtime")
        pct_maxdb = parser.getfloat("pile", "pct_maxdb")
        hydr_sens = parser.getfloat("pile", "hydr_sens")
    else:
        print("monitor.cfg not found. Using default values.")
        minpeakdb = 150
        interval_between_strikes = 0.75
        lookaheadtime = 0.050
        pct_maxdb = 0.0
        hydr_sens = 199.0000

    # calibration_dict = {48000: 96.87013395, 96000: 387.4805358}
    # calibration_dict = {48000: 102.8939249, 96000: 387.4805358}

    run_settings = {
        "minpeakdb": minpeakdb,
        "pct_maxdb": pct_maxdb,
        "lookaheadtime": lookaheadtime,
        "interval_between_strikes": interval_between_strikes,
        "hydr_sens": hydr_sens,
    }

    for i in run_settings:
        print(f"{i}: {run_settings[i]}")

    print("")

    filename_minpeakdb_dict = {}
    for subdir, dirs, files in os.walk(rootdir):
        # dirs are contained in subdir; files in dirs
        files_imp_vib_wav = []
        for file in files:
            if fnmatch.fnmatch(file, "*.wav"):
                if file in already_processed:
                    print(f"{file} already processed.")
                else:
                    # previously fnmatch.fnmatch(file, '*imp*.wav') or fnmatch.fnmatch(file, '*vib*.wav'):
                    files_imp_vib_wav.append(os.path.join(subdir, file))
        if files_imp_vib_wav:
            n1 = dt.datetime.now()
            # to_excel expects full filenames with paths!! Otherwise it throws exception 13.
            # fn = r'C:\Users\Licensed User\PycharmProjects\pilestrike_analysis\recording_2018_08_10_120202.wav'

            # process_files([fn], minpeakdb, pct_maxdb, interval_between_strikes, lookaheadtime, run_settings)  # xxx
            # return files_imp_vib_wav
            dfpeaks_raw, dfpeaks, report_name = process_files(
                files_imp_vib_wav,
                minpeakdb,
                pct_maxdb,
                interval_between_strikes,
                hydr_sens,
                run_settings,
                filename_minpeakdb_dict=filename_minpeakdb_dict,
                str_signal=str_signal,
                progress_signal=progress_signal,
            )  # xxx
            n2 = dt.datetime.now()
            print("process time ", (n2 - n1).microseconds / 1000, "seconds")
            if not all:
                break  # stop recursion for pile monitor
            else:
                print("No files to process")
                dfpeaks_raw = pd.DataFrame()
                dfpeaks = pd.DataFrame()

    return [run_settings, dfpeaks_raw, dfpeaks, report_name]
