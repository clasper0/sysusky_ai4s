class MolecularGCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 36,  # 原子特征维度 (修正为实际的36维)
        hidden_dims: List[int] = [128, 256, 512],  # 隐藏层维度
        output_dim: int = 1,  # 输出维度 (pIC50值)
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        use_residual: bool = True,
        attention_heads: int = 4
    ):
        """
        Initialize the Molecular GCN model.

        Args:
            input_dim: Dimension of input atom features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (prediction target)
            dropout_rate: Dropout rate for regularization
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            attention_heads: Number of attention heads for GAT layers
        """
        super(MolecularGCN, self).__init__()
    #设置模型的输入输出维度、dropout的比例
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims#一个数组
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu





        # Build GCN layers
        #图卷积层
        self.gcn_layers = nn.ModuleList()
        #图注意力层
        self.gat_layers = nn.ModuleList()

        # Input layer输入层
        #输入维度->第一个Hidden dim
        self.gcn_layers.append(GCNConv(input_dim, hidden_dims[0]))
        self.gat_layers.append(GATConv(hidden_dims[0], hidden_dims[0] // attention_heads,
                                       heads=attention_heads, dropout=dropout_rate))
    
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.gcn_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            self.gat_layers.append(GATConv(hidden_dims[i], hidden_dims[i] // attention_heads,
                                          heads=attention_heads, dropout=dropout_rate))

        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            for dim in hidden_dims:
                self.batch_norms.append(nn.BatchNorm1d(dim))

        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_dims[-1])

        # Prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 3, hidden_dims[-1] // 2),  # 3 pooling types concatenated
            nn.Dropout(dropout_rate),
            self._get_activation_layer(),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_activation_layer(self):
        """Get activation layer based on configuration."""
        if isinstance(self.activation, type(F.relu)):
            return nn.ReLU()
        elif isinstance(self.activation, type(F.gelu)):
            return nn.GELU()
        elif isinstance(self.activation, type(F.leaky_relu)):
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()
        

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)