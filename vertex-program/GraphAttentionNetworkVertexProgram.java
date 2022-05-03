/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.tinkerpop.gremlin.process.computer.neuralnetwork;

import org.apache.commons.configuration2.Configuration;
import org.apache.tinkerpop.gremlin.process.computer.*;
import org.apache.tinkerpop.gremlin.process.computer.util.AbstractVertexProgramBuilder;
import org.apache.tinkerpop.gremlin.process.computer.util.StaticVertexProgram;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;
import org.apache.tinkerpop.gremlin.structure.Graph;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.structure.VertexProperty;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class GraphAttentionNetworkVertexProgram extends StaticVertexProgram {

    private static final String FEATURE = "gremlin.GraphAttentionNetworkVertexProgram.property";
    private static final String LABEL = "gremlin.GraphAttentionNetworkVertexProgram.label";
    private static final String PRED = "gremlin.GraphAttentionNetworkVertexProgram.pred";

    private MessageScope.Local<Double> incidentMessageScope = MessageScope.Local.of(__::outE);
    private Set<VertexComputeKey> vertexComputeKeys;
    private String featurePropName;
    private String labelPropName;
    private String predPropName;
    private int featDim;
    private int headDim;
    private int msgDim;
    private int outputDim;
    private float[][][] projectionWeights;
    private float[][][] residualWeights;
    private float[][] A1;
    private float[][] A2;

    @Override
    public void loadState(Graph graph, Configuration configuration) {
        this.featurePropName = configuration.getString(FEATURE, "feature");
        this.labelPropName = configuration.getString(LABEL, "label");
        this.predPropName = configuration.getString(PRED, "pred");
        this.vertexComputeKeys = new HashSet<>(Arrays.asList(
            VertexComputeKey.of(this.featurePropName, true),
            VertexComputeKey.of(this.predPropName, false)));
    }

    @Override
    public void workerIterationStart(final Memory memory) {
        int layer = memory.getIteration() / 2 + 1;
        String featureFilePath = System.getProperty("user.home") + "/Parallel-GAT/models/gat_ppi_model_layer" + layer + ".txt";
        switch (layer) {
            case 1:
                featDim = 50;
                headDim = 4;
                msgDim = 64;
                break;
            case 2:
                featDim = 256;
                headDim = 4;
                msgDim = 64;
                break;
            case 3:
                featDim = 256;
                headDim = 6;
                msgDim = 121;
                break;
        }
        outputDim = headDim * msgDim;
        projectionWeights = new float[headDim][msgDim][featDim];
        residualWeights = new float[headDim][msgDim][featDim];
        A1 = new float[headDim][msgDim];
        A2 = new float[headDim][msgDim];
        try {
            Scanner scanner = new Scanner(new File(featureFilePath));
            for (int k = 0; k < headDim; k++) {
                for (int i = 0; i < msgDim; i++) {
                    A2[k][i] = scanner.nextFloat();
                }
            }
            for (int k = 0; k < headDim; k++) {
                for (int i = 0; i < msgDim; i++) {
                    A1[k][i] = scanner.nextFloat();
                }
            }
            for (int k = 0; k < headDim; k++) {
                for (int i = 0; i < msgDim; i++) {
                    for (int j = 0; j < featDim; j++) {
                        projectionWeights[k][i][j] = scanner.nextFloat();
                    }
                }
            }
            for (int k = 0; k < headDim; k++) {
                for (int i = 0; i < msgDim; i++) {
                    for (int j = 0; j < featDim; j++) {
                        residualWeights[k][i][j] = scanner.nextFloat();
                    }
                }
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void storeState(Configuration configuration) {
        super.storeState(configuration);
        configuration.setProperty(FEATURE, this.featurePropName);
        configuration.setProperty(LABEL, this.labelPropName);
        configuration.setProperty(PRED, this.predPropName);
    }

    @Override
    public void setup(Memory memory) {

    }

    @Override
    public void execute(Vertex vertex, Messenger messenger, Memory memory) {
        float[] inputs = loadInputs(vertex, memory);
        // prepare message
        float[][] msgs = new float[headDim][msgDim];
        for (int k = 0; k < headDim; k++) {
            for (int i = 0; i < msgDim; i++) {
                for (int j = 0; j < featDim; j++) {
                    msgs[k][i] += projectionWeights[k][i][j] * inputs[j];
                }
            }
        }

        if (memory.getIteration() % 2 == 0) {
            // in even iterations, send messages to neighbors
            messenger.sendMessage(this.incidentMessageScope, msgs);
            return;
        }
        // in odd iterations, receive messages from neighbors and calculate output
        float[] heats1 = new float[headDim];
        for (int k = 0; k < headDim; k++) {
            float heat1 = 0;
            for (int i = 0; i < msgDim; i++) {
                heat1 += msgs[k][i] * A1[k][i];
            }
            heats1[k] = heat1;
        }

        float[] maxAffinities = new float[headDim];
        List<float[][]> neighborsMsgs = new ArrayList<>();
        List<float[]> affinitiesList = new ArrayList<>();
        Iterator iter = messenger.receiveMessages();
        while (iter.hasNext()) {
            float[][] neighborMsgs = (float[][]) iter.next();
            neighborsMsgs.add(neighborMsgs);
            // calculate heats for this neighbor
            float[] affinities = new float[headDim];
            for (int k = 0; k < headDim; k++) {
                float heat2 = 0;
                for (int i = 0; i < msgDim; i++) {
                    heat2 += neighborMsgs[k][i] * A2[k][i];
                }
                float affinity = leakyRelu(heats1[k] + heat2);
                maxAffinities[k] = Math.max(maxAffinities[k], affinity);
                affinities[k] = affinity;
            }
            affinitiesList.add(affinities);
        }

        float[] outputs = new float[outputDim];
        for (int k = 0; k < headDim; k++) {
            float sumAffinity = 0;
            for (int i = 0, neighbors = neighborsMsgs.size(); i < neighbors; i++) {
                float[] affinities = affinitiesList.get(i);
                float affinity = (float) Math.exp(affinities[k] - maxAffinities[k]);
                affinities[k] = affinity;
                sumAffinity += affinity;
            }
            for (int i = 0, neighbors = neighborsMsgs.size(); i < neighbors; i++) {
                float[][] neighborMsgs = neighborsMsgs.get(i);
                float[] affinities = affinitiesList.get(i);
                float w = affinities[k] / (sumAffinity + 1e-16f);
                for (int j = 0; j < msgDim; j++) {
                    outputs[k * msgDim + j] += neighborMsgs[k][j] * w;
                }
            }
        }

        // Add skip or residual connection
        if (featDim == msgDim) {
            for (int k = 0; k < headDim; k++) {
                for (int j = 0; j < msgDim; j++) {
                    outputs[k * msgDim + j] += inputs[j];
                }
            }
        } else {
            for (int k = 0; k < headDim; k++) {
                for (int i = 0; i < msgDim; i++) {
                    for (int j = 0; j < featDim; j++) {
                        outputs[k * msgDim + i] += residualWeights[k][i][j] * inputs[j];
                    }
                }
            }
        }

        if (this.terminate(memory)) {
            for (int i = 0; i < msgDim; i++) {
                float avgOutput = 0.f;
                for (int k = 0; k < headDim; k++) {
                    avgOutput += outputs[k * msgDim + i];
                }
                vertex.property(VertexProperty.Cardinality.list, this.predPropName, avgOutput >= 0 ? 1 : 0);
            }
        } else {
            // remove existing old intermediate results
            Iterator<VertexProperty<Double>> propIter = vertex.properties(this.featurePropName);
            while (propIter.hasNext()) {
                propIter.next().remove();
            }
            // activate and save intermediate results as the input for next layer
            for (int i = 0; i < outputDim; i++) {
                float output = outputs[i];
                if (output <= 0) {
                    output = (float) Math.exp(output) - 1;
                }
                vertex.property(VertexProperty.Cardinality.list, this.featurePropName, output);
            }
        }
    }

    private float leakyRelu(float value) {
        if (value < 0) {
            return 0.2f * value;
        }
        return value;
    }

    @Override
    public Set<VertexComputeKey> getVertexComputeKeys() {
        return this.vertexComputeKeys;
    }

    @Override
    public boolean terminate(Memory memory) {
        return memory.getIteration() == 5;
    }

    @Override
    public Set<MessageScope> getMessageScopes(Memory memory) {
        final Set<MessageScope> set = new HashSet<>();
        set.add(this.incidentMessageScope);
        return set;
    }

    @Override
    public GraphComputer.ResultGraph getPreferredResultGraph() {
        return null;
    }

    @Override
    public GraphComputer.Persist getPreferredPersist() {
        return null;
    }

    private float[] loadInputs(Vertex vertex, Memory memory) {
        float[] inputs = new float[featDim];
        Iterator<VertexProperty<Float>> inputIter = vertex.properties(this.featurePropName);
        if (inputIter.hasNext()) {
            // Intermediate results are stored in a single property with list cardinality
            for (int i = 0; i < featDim; i++) {
                inputs[i] = inputIter.next().value();
            }
        } else {
            // Due to a minor issue with TinkerPop-Kryo integration, the input graph has to store the raw features
            // in separate properties.
            for (int i = 0; i < featDim; i++) {
                inputs[i] = ((Double) vertex.property(this.featurePropName + i).value()).floatValue();
            }
        }
        return inputs;
    }

    public static GraphAttentionNetworkVertexProgram.Builder build() {
        return new GraphAttentionNetworkVertexProgram.Builder();
    }

    public final static class Builder extends AbstractVertexProgramBuilder<GraphAttentionNetworkVertexProgram.Builder> {

        private Builder() {
            super(GraphAttentionNetworkVertexProgram.class);
        }

        public GraphAttentionNetworkVertexProgram.Builder feature(final String feature) {
            this.configuration.setProperty(FEATURE, feature);
            return this;
        }

        public GraphAttentionNetworkVertexProgram.Builder label(final String label) {
            this.configuration.setProperty(LABEL, label);
            return this;
        }

        public GraphAttentionNetworkVertexProgram.Builder pred(final String label) {
            this.configuration.setProperty(LABEL, label);
            return this;
        }
    }
}
