import pandas as pd

# Original data
data = [
    (2, 'no leak'), 
    (4, '2% leak at 0 km'),
    (12, 'no leak'),
    (4, '4% leak at 0 km'),
    (12, 'no leak'),
    (4, '2% leak at 130 km'),
    (12, 'no leak'),
    (4, '4% leak at 130 km'),
    (12, 'no leak'),
    (4, '2% leak at 250 km'),
    (12, 'no leak'),
    (4, '4% leak at 250 km'),
    (12, 'no leak'),
    (4, '10% leak at 0 km'),
    (18, 'no leak')
]

state_to_fault = {
    'no leak': 'no_fault',
    '2% leak at 0 km': '2pct0',
    '4% leak at 0 km': '4pct0',
    '2% leak at 130 km': '2pct130',
    '4% leak at 130 km': '4pct130',
    '2% leak at 250 km': '2pct250',
    '4% leak at 250 km': '4pct250',
    '10% leak at 0 km': '10pct0'
}

#load the exsting data
df = pd.read_csv(r'G:\GlitchDetect\data\long_run.csv')
# add a new col state and fault_label
df['State'] = None
df['Fault_Label'] = None

i = 0
for hrs, state in data:
    duration = hrs * 36 * 20
    df.loc[i:i + duration - 1, 'State'] = state
    df.loc[i:i + duration - 1, 'Fault_Label'] = state_to_fault[state]
    i = i + duration + 1




#save the updated df
df.to_csv(r'G:\GlitchDetect\data\long_run_labeled.csv', index=False)
