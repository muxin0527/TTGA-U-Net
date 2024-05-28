import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv, GATv2Conv, GraphConv


class GraphAttention(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        ### output projection ###
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        ### output projection ###
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class TopKPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(TopKPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = nn.Linear(in_channels, 1)  # Transform node features to scores

    def forward(self, graph, node_features):
        scores = self.score_layer(node_features).squeeze()  # Compute scores (N,)
        num_nodes = graph.number_of_nodes()
        k = int(self.ratio * num_nodes)

        _, idx = torch.topk(scores, k)

        # 创建子图，并返回池化后的图和特征
        subgraph = graph.subgraph(idx)
        subgraph.ndata['h'] = node_features[idx]

        return subgraph, subgraph.ndata['h'], idx


class GraphUnpooling(nn.Module):
    def __init__(self):
        super(GraphUnpooling, self).__init__()

    def forward(self, features, idx, total_nodes):
        # 创建一个空的特征张量，用于恢复原始图的大小
        unpool_features = torch.zeros(total_nodes, features.size(1))
        # 将特征放回原来的节点索引位置
        unpool_features[idx] = features
        return unpool_features


class GCN(nn.Module):
    def __init__(self, g, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.g = g
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, features):
        x = F.relu(self.conv1(self.g, features))
        x = self.conv2(self.g, x)
        return x


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)

        return h


class GCT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCT, self).__init__()

        self.MLPc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.MLPc2 = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        inputs = self.MLPc1(inputs)

        # 最大池化和平均池化操作
        max_pooled = torch.max(inputs, dim=1, keepdim=True)[0]
        avg_pooled = torch.mean(inputs, dim=1, keepdim=True)

        # 使用池化的结果对输入特征进行缩放（注意力机制）
        attention_max = F.sigmoid(max_pooled)
        attention_avg = F.sigmoid(avg_pooled)

        # 将注意力加权的特征与原始特征相乘
        refined_features_max = attention_max * inputs
        refined_features_avg = attention_avg * inputs

        # 合并两个通道的特征
        combined_features = refined_features_max + refined_features_avg

        # 第二个MLP
        outputs = self.MLPc2(combined_features.squeeze(1))

        return outputs


class FeatureFusionModule(nn.Module):
    def __init__(self, spatial_features_dim, channel_features_dim, output_dim):
        super(FeatureFusionModule, self).__init__()

        # For spatial features transformation
        self.conv1x3 = nn.Conv1d(spatial_features_dim, spatial_features_dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv1d(spatial_features_dim, spatial_features_dim, kernel_size=(3, 1), padding=(1, 0))

        # For channel features transformation
        self.mlp_channel = nn.Sequential(
            nn.Linear(channel_features_dim, spatial_features_dim),  # Transform to match spatial feature dimensions
            nn.ReLU(),
            nn.Linear(spatial_features_dim, spatial_features_dim)
        )

        # Fusion MLP for combining all features into final output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * spatial_features_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, spatial_features, channel_features):
        # Transform spatial features
        spatial_features = spatial_features.unsqueeze(2)  # Adding an extra dimension for Conv1d compatibility
        F_s1x3 = self.conv1x3(spatial_features).squeeze(2)
        F_s3x1 = self.conv3x1(spatial_features).squeeze(2)

        # Transform channel features
        F_c_prime = self.mlp_channel(channel_features)

        # Concatenate spatial and channel features
        F_s1x3 = torch.cat((F_s1x3, F_c_prime), dim=1)
        F_s3x1 = torch.cat((F_s3x1, F_c_prime), dim=1)

        out_features = F_s1x3 * F_s3x1

        # Fusion of features and final output
        out_features = self.fusion_mlp(out_features)
        return out_features


# Define the TGA Encoder
class TGAEncoder(nn.Module):
    def __init__(self, g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim, gct_output_dim,
                 gat_features_dim, gct_features_dim, ff_output_dim,
                 att_hidden_dim,
                 topk_input, pooling_ratio):
        super(TGAEncoder, self).__init__()
        self.spatial = GAT(g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual)
        self.channel = GCT(gct_input_dim, gct_hidden_dim, gct_output_dim)
        self.fusion = FeatureFusionModule(gat_features_dim, gct_features_dim, ff_output_dim)
        self.MLPatt = nn.Sequential(
            nn.Linear(gct_output_dim, att_hidden_dim),
            nn.ReLU(),
            nn.Linear(att_hidden_dim, 2 * gct_output_dim)
        )
        self.pool = TopKPooling(topk_input, ratio=pooling_ratio)

    def forward(self, graph, features):
        spatial_features = self.spatial(features)
        channel_features = self.channel(features)
        fused_features = self.fusion(spatial_features, channel_features)
        con_features = torch.cat((spatial_features, channel_features), dim=1)
        att_features = self.MLPatt(con_features)
        att_features = con_features * att_features
        pooled_graph, pooled_features, idx = self.pool(graph, att_features)
        return pooled_graph, pooled_features, fused_features, idx


# Define the TGA Decoder
class TGADecoder(nn.Module):
    def __init__(self, g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim, gct_output_dim,
                 att_hidden_dim):
        super(TGADecoder, self).__init__()
        self.spatial = GAT(g,
                           num_layers,
                           in_dim,
                           num_hidden,
                           heads,
                           activation,
                           feat_drop,
                           attn_drop,
                           negative_slope,
                           residual)
        self.channel = GCT(gct_input_dim, gct_hidden_dim, gct_output_dim)
        self.MLPatt = nn.Sequential(
            nn.Linear(gct_output_dim, att_hidden_dim),
            nn.ReLU(),
            nn.Linear(att_hidden_dim, 2 * gct_output_dim)
        )
        self.unpool = GraphUnpooling()

    def forward(self, graph, features, fused_features, pool_idx):
        features = self.unpool(features, pool_idx, graph.number_of_nodes())
        features = features + fused_features
        spatial_features = self.spatial(features)
        channel_features = self.channel(features)
        con_features = torch.cat((spatial_features, channel_features), dim=1)
        att_features = self.MLPatt(con_features)
        att_features = con_features * att_features

        return graph, att_features


# Define the Graph U-Net incorporating three encoders and decoders
class TGA_UNet(nn.Module):
    def __init__(self, g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim, gct_output_dim,
                 gat_features_dim, gct_features_dim, ff_output_dim,
                 att_hidden_dim,
                 topk_input, pooling_ratio,
                 gcn_input_dim, gcn_hidden_dim, num_classes):
        super(TGA_UNet, self).__init__()
        self.encoder1 = TGAEncoder(g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim, gct_output_dim,
                 gat_features_dim, gct_features_dim, ff_output_dim,
                 att_hidden_dim,
                 topk_input, pooling_ratio * 0.9)
        self.encoder2 = TGAEncoder(g,
                 num_layers,
                 in_dim,
                 num_hidden * 2,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim * 2, gct_output_dim * 2,
                 gat_features_dim * 2, gct_features_dim * 2, ff_output_dim * 2,
                 att_hidden_dim * 2,
                 topk_input, pooling_ratio * 0.7)
        self.encoder3 = TGAEncoder(g,
                 num_layers,
                 in_dim * 2,
                 num_hidden * 4,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim * 2, gct_hidden_dim * 4, gct_output_dim * 4,
                 gat_features_dim * 4, gct_features_dim * 4, ff_output_dim * 4,
                 att_hidden_dim * 4,
                 topk_input, pooling_ratio * 0.5)

        self.decoder1 = TGADecoder(g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim, gct_output_dim,
                 att_hidden_dim)
        self.decoder2 = TGADecoder(g,
                 num_layers,
                 in_dim,
                 num_hidden * 2,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim, gct_hidden_dim * 2, gct_output_dim * 2,
                 att_hidden_dim * 2)
        self.decoder3 = TGADecoder(g,
                 num_layers,
                 in_dim * 2,
                 num_hidden * 4,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 gct_input_dim * 2, gct_hidden_dim * 4, gct_output_dim * 4,
                 att_hidden_dim * 4)
        self.gcn = GCN(g, gcn_input_dim, gcn_hidden_dim, num_classes)

    def forward(self, graph, features):
        # Encode
        feats = []
        graph, features, fused_feature1, idx1 = self.encoder1(graph, features)
        feats.append(features)
        graph, features, fused_feature2, idx2 = self.encoder2(graph, features)
        feats.append(features)
        graph, features, fused_feature3, idx3 = self.encoder3(graph, features)
        feats.append(features)
        # Decode
        graph, features = self.decoder1(graph, torch.cat((features, feats[-1]), dim=1), fused_feature1, idx1)
        graph, features = self.decoder2(graph, torch.cat((features, feats[-2]), dim=1), fused_feature2, idx2)
        graph, features = self.decoder3(graph, torch.cat((features, feats[-3]), dim=1), fused_feature3, idx3)
        output = self.gcn(features)

        return output


class GATv2(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATv2, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATv2Conv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATv2Conv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATv2Conv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
