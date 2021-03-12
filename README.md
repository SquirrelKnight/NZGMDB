# ESGMIM
The NZ earthquake and ground motion database includes important earthquake data, such as hypocenter locations, corrected local magnitudes, and ground motion intensity measures. This database has been designed to be easily expanded. These pages will break down each catalogue that make up the database, and will outline procedures used to generate the catalogues.

Welcome to ver. 0.4 of the National Seismic Hazard Model Earthquake Source and Ground 
Motion Intensity Measure catalogue (there is currently no clever name, sadly)! Here you 
will find several tables, which include data for M >= 4 data (from the original GeoNet 
database) with high quality ground motion intensity measures:

	-Earthquake Source Table
	-Site Table
	-Station Magnitude Table
	-Phase Arrival Table
	-Propagation Path Table
	-Earthquake Ground Motion Intensity Measure Catalogue
	
These tables can easily be read into almost any programs, as they are CSV files. They
were created using the Pandas module for Python.

More information on the tables can be found on our in-progress Wiki! All relevant tables
are linked to from the Related Resources category below the table descriptions:

https://wiki.canterbury.ac.nz/display/QuakeCore/Ground+Motion+Intensity+Measure+Catalogue

Current Version Implementations:
- v 0.4 - The zip file has been organized to have two folders, Figures and Tables.
	Additional columns have been added to the GM IM table. These include:
	'ev_lat', 'ev_lon', 'ev_depth', 'mag', 'tect_type', 'reloc', 'sta_lat', 'sta_lon',
	'Vs30', 'Tsite', 'Z1.0', 'Z2.5', 'rrup', 'rjb', and 'Ztor'. Please note that
	rrup, rjb, and Ztor are currently duplicates of r_hyp, r_epi from the propagation
	path table, and event depth.

Previous Version Information:
- v 0.3 - Accompanying figures have been added to the ESGMIM package in .png format.
- v 0.2 - The GM IM catalogue has been divided into three separate documents for 000, 090,
	and vertical components. The site table contains additional site response data.
- v 0.1 - Includes complete subset catalogue with GM IM and relocation data for M >= 4 
	GM IM data. As includes tectonic class measurements.
- v 0.0 - First version. Incomplete and without the addition of relocated data.


Earthquake Source Table:

	evid | Event identification number.
	datetime | Origin time of the event.
	lat, lon, depth | Origin latitude and longitude of the event in decimal degrees, with 
		depth in km below sea level.
	loc_type, loc_grid | Location method or catalogue used for the event and when 
		applicable, the location grid used for location.
	mag, mag_type, mag_method, mag_unc | The magnitude (determined from the median of the 
		station magnitudes) and magnitude type of the event (direct MW vs ML corrected to
		 MW) and method used to calculate it, as well as the uncertainty.
	tect_type, tect_method | The tectonic type of the event (crustal, interface, or slab) 
		and the method used to determine it
	ndef, nsta | The number of picks and stations used in locating the earthquake
	t_res, precision | When applicable, the origin time residual in seconds and the 
		decimal precision of the hypocenter in kilometers.
	relocated | Whether the associated event has been relocated (yes or no).

Site Table:

	net, sta | Network and station name.
	lat, lon | Station latitude and longitude coordinate in decimal degrees.
	elev | Station elevation, in meters above sea level.
	site_class | Site classification by rank (A is best, D is worst)
	Vs30 | 30-m averaged shear-wave velocity (m/s).
	Tsite | Fundamental site periods (s)
	Z1.0, Z2.5 | Depths (in km) to shear-wave velocities of 1.0 and 2.5 km/s, 
		respectively.
	Q_Vs30 | Quality rating of Vs30 measurement (see Kaiser et al. (2017))
	Q_Tsite | Quality rating of Tsite measurement (see Kaiser et al. (2017))
	D_Tsite | Method used for determining site period (see Kaiser et al. (2017))
	Q_Z1.0 | Quality rating of Z1.0 measurement (see Kaiser et al. (2017))

Station Magnitude Table:

	magid | Magnitude identification number.
	net, sta, z_chan, e_chan, n_chan | Network, station, and 3-component channels used to 
		determine magnitude.
	evid | Event identification number associated with the source-receiver pair.
	mag | Uncorrected magnitude.
	mag_type | Magnitude type of the uncorrected magnitude.
	mag_corr | Corrected local magnitude.
	mag_corr_method | Method used to correct local magnitude. Currently using equations 
		from Rhoades et al. (2020).
	amp | Peak amplitude of the signal in mm. Some uncorrected data is only reported in
		counts.
	relocated | Whether the associated event has been relocated (yes or no).

Phase Arrival Table:

	arid | Arrival identification number.
	datetime | Time of phase arrival.
	net, sta, chan | Seismic network, station, and channel where the arrival was detected.
	phase | Phase of the arrival (e.g., P- or S-phases).

Propagation Path Table:

	evid | Event identification number.
	net, sta | Network and station name.
	r_epi, r_hyp | Epicentral and hypocentral distances between station and event (km).
	az, b_az | Azimuth from the event to the station and back-azimuth from the station to 
		the event.
	toa | Takeoff angle from the event to the station. Currently all toas are derived from 
		those reported in the GeoNet database.
	relocated | Whether the associated event has been relocated (yes or no).
	rrup, rjb | Shortest distance to the rupture plane and Joyner-Boore distance, the 
		shortest distance to the surface projection of a rupture plane (in km). These are 
		roughly equivalent to r_hyp and r_epi, respectively.

Ground Motion Intensity Measure Catalogue:

	gmid | Ground motion identification number.
	evid | Event identification number.
	net, sta, loc | Network, location (number associated to different instruments at a site/station), and station name.
	PGA, PGV | Peak ground acceleration (g) and velocity (cm/s).
	SA(T) | Pseudo-acceleration response spectra (spectral acceleration, g) for various periods, T.
	CAV, AI | Cumulative absolute velocity (g*s) and Arias intensity (m/s).
	Ds575, Ds595 | Significant duration (s) for different energy thresholds (5-75% and 5-95%).
	score_X, score_Y, score_Z | Classification score (generally from 0-1) on the quality of the GM for X, Y, and Z channels.
	f_min_X, f_min_Y, f_min_Z | Minimum viable frequency (Hz) of the GM for the X, Y, and Z components.
	ev_lat, ev_lon, ev_depth, mag, tect_type | Event latitude, longitude, depth (km), magnitude, and tectonic type.
	Vs30 | 30-m averaged shear-wave velocity (m/s).
	Tsite | Fundamental site periods (s).
	Z1.0, Z2.5 | Depths (in km) to shear-wave velocities of 1.0 and 2.5 km/s, respectively.
	Rrup, Rjb | Shortest distance to the rupture plane and Joyner-Boore distance, the shortest distance to the surface projection of a rupture plane (in km). These are roughly equivalent to r_hyp and r_epi, respectively.
	Ztor | Depth to the top of the rupture (in km).
  
