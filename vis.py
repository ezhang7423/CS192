import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=12, ncols=3, figsize=(30, 100))


def getAvg(df):
    res = pd.Series(np.zeros(128))
    for x in df:
        res += df[x]
    res /= (len(list(dfH)) - 1)
    return res


subject_nums = [
    '04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16'
]
conditions = ['bikeHigh', 'bikeLow', 'resting']
typeimgs = ['target', 'distractor']
handpicked = [0, 1, 2, 39, 28, 29]
p3andfriends = [8, 21, 12, 11, 18, 13, 17]


def getAvgg(subj, typeimg, condition, r, c):
    dat = store[condition][typeimg][subj]
    totalDf = pd.DataFrame()
    for x in range(dat.shape[0]):
        df = pd.DataFrame(dat[x]).transpose()
        df['avg'] = getAvg(df[p3andfriends])
        totalDf[x] = getAvg(df[p3andfriends])
    ax = getAvg(totalDf).plot(title=f"Avg of {subj} during {condition}",
                              ax=axes[r, c],
                              label=str(typeimg))
    ax.legend(loc="upper left")
    ax.set(xlabel=f"Time (ms)", ylabel=f"Voltage (?)")
    #ax.text(65, -200, f"average over channels {p3andfriends} across all trials", ha='center')
    print(f'finished {condition} {typeimg} {subj}')


for i, x in enumerate(subject_nums):
    for y in typeimgs:
        for j, z in enumerate(conditions):
            getAvgg(x, y, z, i, j)
# dfH = df.transpose()[handpicked]
# dfP = df.transpose()[p3andfriends]
# getAvg(dfH)['avg'].plot(legend=False, figsize=(15, 8))
# df.plot()
