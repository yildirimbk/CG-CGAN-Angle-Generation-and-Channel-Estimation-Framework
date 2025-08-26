import numpy as np
                                            #Adjustable parameters:
def get_adjustable_parameters(args,parameters):
    #1. Set dynamic scenario Scenes
    parameters['dynamic_scenario_scenes'] = np.arange(args.dynamic_scenario_scenes[0]-1,args.dynamic_scenario_scenes[1])

    #2. Max number of strongest paths
    parameters['num_paths'] = args.num_paths

    ##3. Activating BSs
    parameters['active_BS'] = np.array(args.active_BS)

    ##4. Select User Rows
    parameters['user_rows'] = np.arange(args.user_rows[0]-1,args.user_rows[1])

    ##5. Select User Antenna Shape
    parameters['ue_antenna']['shape'] = np.array(args.user_antenna_shape)

    ##6. Select User Antenna Spacing
    parameters['ue_antenna']['spacing'] = args.user_antenna_spacing

    ##7. Define user antenna rotation
    parameters['ue_antenna']['rotation'] = np.array(args.user_antenna_rotation)

    ##8. Define user antenna FoV
    parameters['ue_antenna']['FoV'] = np.array(args.user_antenna_FoV)

    ##9. Define user antenna radiation pattern
    parameters['ue_antenna']['radiation_pattern'] = args.user_antenna_radiationpattern

    #10. Select BS Antenna Shape
    parameters['bs_antenna']['shape'] = np.array(args.BS_antenna_shape)

    #11. Select BS Antenna Spacing
    parameters['bs_antenna']['spacing'] = args.BS_antenna_spacing

    #12. Define BS antenna rotation
    parameters['bs_antenna']['rotation'] = np.array(args.BS_antenna_rotation)

    #13. Define BS antenna FoV
    parameters['bs_antenna']['FoV'] = np.array(args.BS_antenna_FoV)

    #14. Define BS antenna radiation pattern
    parameters['bs_antenna']['radiation_pattern'] = args.BS_antenna_radiationpattern

    ## Different antenna configuration for different BSs

    ## Consider 3 active basestations
    #parameters['active_BS'] = np.array(args.active_BS)

    ## Define 3 different antennas:
    #antenna1 = {'shape': np.array([1, 1]),
    #            'spacing': 0.5,
    #            'rotation': np.array([0, 30, 0])}
    #antenna2 = {'shape': np.array([2, 2]),
    #            'spacing': 0.5,
    #            'rotation': np.array([-15, 0, 30])}
    #antenna3 = {'shape': np.array([3, 4]),
    #            'spacing': 0.5,
    #            'rotation': np.array([-15, 0, 0])}
    # Assign the defined antennas to the active basestations:
    #parameters['bs_antenna'] = [antenna1, antenna2, antenna3]

    ##15. To generate basestation to basestation output variables
    parameters['enable_BS2BS'] = args.enable_BS2BS

    ##16. Decide generation of the channels with the Doppler shift
    parameters['enable_doppler'] = args.enable_doppler

    ##17. To generate  dual polar antennas
    parameters['enable_dual_polar'] = args.enable_dualpolar

    ##18. To activate/deactivate OFDM system
    parameters['activate_OFDM'] = args.activate_OFDM

    ##19. To set OFDM channel bandwidth
    parameters['OFDM']['bandwidth'] = args.OFDM_bandwidth

    ##20. To set the number of OFDM subcarriers
    parameters['OFDM']['subcarriers']=args.OFDM_subcarriers

    ##21. Select the desired subcarrier indices
    parameters['OFDM']['selected_subcarriers']=args.OFDM_selectedsubcarriers

    ##22. Decide whether having a receive LPF filter
    parameters['OFDM']['RX_filter']=args.OFDM_Rxfilter

    ## Print parameters for double check

    #for key, value in parameters.items():
    #    print(f"{key}: {value}")
    return parameters
