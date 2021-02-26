import tensorflow as tf
import numpy as np

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model, Transformer
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import *

from keras.layers import Input, Dense, Lambda, Activation


# 修改参数路径
pretrain_model_save_path = "./pretrain_weights/chinese_t5s_tiny/"


class T5SEncoder(Transformer):
    """T5模型（Encoder）+ 类型编码
    """
    
    def __init__(
        self,
        segment_vocab_size=2,  # segment总数目
        with_mlm=False,  # 是否包含MLM部分
        **kwargs  # 其余参数
    ):
        super(T5SEncoder, self).__init__(**kwargs)
        self.segment_vocab_size = segment_vocab_size
        self.with_mlm = with_mlm
    
    def get_inputs(self):
        """T5的Encoder的输入只有token_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Input-Token'
        )
        
        inputs = [x_in]
        
        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)
        
        return inputs

    def apply_embeddings(self, inputs):
        """T5的token embedding + segment embedding，
        并把relative position embedding准备好，待attention使用。
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
            
        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        
        if self.segment_vocab_size > 0:
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name='Embedding-Segment'
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """T5的Encoder的主体是基于Self-Attention的模块
        顺序：LN --> Att --> Add --> LN --> FFN --> Add
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )
        x = self.apply(
            inputs=[x, x, x, position_bias],
            layer=MultiHeadAttention,
            arguments={'p_bias': 't5_relative'},
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Output-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Output-Dropout'
        )
        
        scale_weight = np.sqrt(self.hidden_size)
        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x: x / scale_weight,
            mask=lambda i, m: m,
            name='Output-Scale'
        )
        
        outputs = [x]
        
        if self.with_mlm:
            
            # 先加上吧，不知道会不会影响
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act[0],
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                center=False,
                epsilon=1e-6,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            
#             # Masked Language Model部分
#             x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)
        
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]
        
        return outputs

    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x = inputs
            p = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=True,
                embeddings_initializer=self.initializer,
                name='Embedding-Relative-Position'
            )
            self.position_bias = p

        return self.position_bias
    
    @classmethod
    def startswith(cls, inputs):
        return False

    
t5s_tokenizer = Tokenizer(pretrain_model_save_path + "vocab.txt")

t5s = build_transformer_model(
    config_path=pretrain_model_save_path + "t5s_config.json",
    model=T5SEncoder,
    # with_mlm='linear', 
    return_keras_model=True,
)

t5s.load_weights(pretrain_model_save_path + "model.h5", by_name=True)