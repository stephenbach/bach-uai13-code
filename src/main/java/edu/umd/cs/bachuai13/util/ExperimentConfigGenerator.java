package edu.umd.cs.bachuai13.util;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.configuration.ConfigurationException;

import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType;
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager;
import edu.umd.cs.psl.reasoner.bool.BooleanMCSatFactory;
import edu.umd.cs.psl.reasoner.bool.BooleanMaxWalkSatFactory;

/**
 * Generates a set of {@link ConfigBundle ConfigBundles}.
 * 
 * @author Stephen Bach <bach@cs.umd.edu>
 */
public class ExperimentConfigGenerator {
	
	public ExperimentConfigGenerator(String baseConfigName) {
		this.baseConfigName = baseConfigName;
	}

	/* General options */
	protected final String baseConfigName;
	
	protected List<String> modelTypes = new ArrayList<String>();
	public void setModelTypes(List<String> modelTypes) {
		this.modelTypes = modelTypes;
	}
	
	protected List<String> learningMethods = new ArrayList<String>();
	public void setLearningMethods(List<String> learningMethods) { this.learningMethods = learningMethods; }
	
	/* VotedPerceptron options */
	protected List<Integer> votedPerceptronStepCounts = new ArrayList<Integer>();
	public void setVotedPerceptronStepCounts(List<Integer> vpStepCounts) {
		votedPerceptronStepCounts = vpStepCounts;
	}
	
	protected List<Double> votedPerceptronStepSizes = new ArrayList<Double>();
	public void setVotedPerceptronStepSizes(List<Double> vpStepSizes) {
		votedPerceptronStepSizes = vpStepSizes;
	}
	
	/* MaxMargin options */
	protected List<Double> maxMarginSlackPenalties = new ArrayList<Double>();
	public void setMaxMarginSlackPenalties(List<Double> mmSlackPenalties) {
		maxMarginSlackPenalties = mmSlackPenalties;
	}
	
	protected List<LossBalancingType> maxMarginLossBalancingTypes = new ArrayList<LossBalancingType>();
	public void setMaxMarginLossBalancingTypes(List<LossBalancingType> mmLossBalancingTypes) {
		maxMarginLossBalancingTypes = mmLossBalancingTypes;
	}
	protected List<NormScalingType> maxMarginNormScalingTypes = new ArrayList<NormScalingType>();
	public void setMaxMarginNormScalingTypes(List<NormScalingType> mmNormScalingTypes) {
		maxMarginNormScalingTypes = mmNormScalingTypes;
	}
	
	protected List<Boolean> maxMarginSquaredSlackValues = new ArrayList<Boolean>();
	public void setMaxMarginSquaredSlackValues(List<Boolean> mmSquaredSlackValues) {
		maxMarginSquaredSlackValues = mmSquaredSlackValues;
	}
	
	public List<ConfigBundle> getConfigs() {
		List<ConfigBundle> configs = new ArrayList<ConfigBundle>();
		ConfigManager cm;
		try {
			cm = ConfigManager.getManager();
		} catch (ConfigurationException e) {
			throw new RuntimeException(e);
		}
		String name;
		
		for (String modelType : modelTypes) {
			for (String learningMethod : learningMethods) {
				if (learningMethod.equals("MLE") || learningMethod.equals("MPLE")) {
					for (int vpStepCount : votedPerceptronStepCounts) {
						for (double vpStepSize : votedPerceptronStepSizes) {
							ConfigBundle newBundle = cm.getBundle(baseConfigName);
							newBundle.addProperty("learningmethod", learningMethod);
							newBundle.addProperty(VotedPerceptron.NUM_STEPS_KEY, vpStepCount);
							newBundle.addProperty(VotedPerceptron.STEP_SIZE_KEY, vpStepSize);
							name = modelType + "-" + learningMethod.toLowerCase() + "-" + vpStepCount + "-" + vpStepSize;
							if (modelType.equals("bool")) {
								newBundle.addProperty(VotedPerceptron.REASONER_KEY, new BooleanMaxWalkSatFactory());
								newBundle.addProperty(MaxPseudoLikelihood.BOOLEAN_KEY, true);
							}
							addInferenceProperties(newBundle, modelType);
							newBundle.addProperty("name", name);
							configs.add(newBundle);
						}
					}
				}
				else if (learningMethod.equals("MM")) {
					for (double slackPenalty : maxMarginSlackPenalties) {
						for (LossBalancingType lossBalancing : maxMarginLossBalancingTypes) {
							for (NormScalingType normScaling : maxMarginNormScalingTypes) {
								for (boolean squareSlack : maxMarginSquaredSlackValues) {
									ConfigBundle newBundle = cm.getBundle("epinions");
									newBundle.addProperty("learningmethod", learningMethod);
									newBundle.addProperty(MaxMargin.SLACK_PENALTY_KEY, slackPenalty);
									newBundle.addProperty(MaxMargin.BALANCE_LOSS_KEY, lossBalancing);
									newBundle.addProperty(MaxMargin.SCALE_NORM_KEY, normScaling);
									newBundle.addProperty(MaxMargin.SQUARE_SLACK_KEY, squareSlack);
									if (modelType.equals("bool")) {
										newBundle.addProperty(MaxMargin.REASONER_KEY, new BooleanMaxWalkSatFactory());
									}
									addInferenceProperties(newBundle, modelType);
									name = modelType + "-mm-" + slackPenalty + "-" + lossBalancing.name().toLowerCase() + "-" + normScaling.name().toLowerCase() + "-" + squareSlack;
									newBundle.addProperty("name", name);
									configs.add(newBundle);
								}
							}
						}
					}
				}
				else {
					throw new IllegalArgumentException("Unrecognized learning method: " + learningMethod);
				}
			}
		}
		
		return configs;
	}
	
	private void addInferenceProperties(ConfigBundle config, String modelType) {
		if (modelType.equals("bool")) {
			config.addProperty(MPEInference.REASONER_KEY, new BooleanMCSatFactory());
			config.addProperty(LazyMPEInference.REASONER_KEY, new BooleanMCSatFactory());
		}
	}
}
