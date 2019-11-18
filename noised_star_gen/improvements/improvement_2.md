# Mis-prediction Error

## Schema

Assumes that the original signal is a sine wave. This might could be
extended later, but for now we will focus on sine waves.

- sine wave amplitude
- sine wave period (in seconds)
- sine wave phase (in degrees)
- percent mis-predicted
- mis-prediction type (phase, amplitude, phase and amplitude)
- percent burst (in events/not points)
- percent single (in events/not points)
- burst length (fixed for now)

-- Errors are inserted using a uniform distributed random variable over the length
   - i.e. uniform(length, #errors)
-- follow a mark position first then insert typed errors pattern

### Calculating the number of each event

We need to change percent events for burst and single into number of each event for marking.
We must also do this under the constrain that only %Err percent of points (not events) is marked.

Givens:
- DataLen: Data length
- %Bev: burst event rate
- %Sev: single event rate
- %Epnts: person of points errored

Derivation:
- #events = #Bev + #Sev
- #Epnts = %Epnts * DataLen
- #Epnts = #Bev * BLen + #Sev (used number of points)
- #Epnts = %Bev * #events * BLen + %Sev * #events
- #events = #Epnts/(%Bev * BLen + %Sev)
- #Bev = %Bev * #events
- #Sev = %Sev * #events

Q.E.D.
